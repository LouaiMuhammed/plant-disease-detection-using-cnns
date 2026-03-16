import argparse
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUGMENT_PREFIXES = (
    "blurred_",
    "brightened_",
    "cropped_",
    "flipped_horizontal_",
    "flipped_vertical_",
    "gamma_corrected_",
    "grayscale_",
    "inverted_",
    "noisy_",
    "rotated_",
    "scaled_",
    "sharpened_",
    "sheared_",
    "shifted_",
    "zoomed_",
)
VARIANT_SUFFIX_RE = re.compile(r"(?:_added\d+)$")
CUSTOM_SUFFIX_RE = re.compile(r"\s*\(Custom\)$", re.IGNORECASE)
IMAGE_PAREN_VARIANT_RE = re.compile(r"^(Image\d+)\((\d+)\)$", re.IGNORECASE)
IMG_UNDERSCORE_VARIANT_RE = re.compile(r"^(IMG_\d{8}_\d{6})_\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class Item:
    path: Path
    class_name: str
    stem: str
    group_id: str
    file_hash: str


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_stem(stem: str) -> str:
    s = stem.strip()
    for prefix in AUGMENT_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = CUSTOM_SUFFIX_RE.sub("", s)
    s = VARIANT_SUFFIX_RE.sub("", s)
    m = IMAGE_PAREN_VARIANT_RE.match(s)
    if m:
        s = m.group(1)
    m = IMG_UNDERSCORE_VARIANT_RE.match(s)
    if m:
        s = m.group(1)
    s = re.sub(r"\s+", " ", s).strip(" _-")
    return s or stem


def group_id_for(class_name: str, stem: str) -> str:
    s = normalize_stem(stem)

    if s.startswith("new_canker_"):
        return f"{class_name}:{s}"

    # Keep IMG timestamps grouped by full timestamp, not generic IMG prefix.
    if s.startswith("IMG_"):
        return f"{class_name}:{s}"

    # Keep YYYYMMDD_* grouped by full timestamp/session id.
    if re.match(r"^\d{8}_\d{6}", s):
        return f"{class_name}:{s}"

    # For names like b (11) or Image00234, collapse only known duplicate suffixes.
    return f"{class_name}:{s}"


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def collect_items(train_dir: Path, val_dir: Path) -> List[Item]:
    items: List[Item] = []
    for split_dir in (train_dir, val_dir):
        for path in iter_images(split_dir):
            class_name = path.parent.name
            stem = path.stem
            items.append(
                Item(
                    path=path,
                    class_name=class_name,
                    stem=stem,
                    group_id=group_id_for(class_name, stem),
                    file_hash=file_hash(path),
                )
            )
    return items


def summarize(items: List[Item]) -> Tuple[Counter, Counter, Counter]:
    class_counts = Counter()
    duplicate_counts = Counter()
    group_counts = Counter()
    seen_hashes: Dict[str, str] = {}
    for item in items:
        class_counts[item.class_name] += 1
        group_counts[item.class_name] += 1
        key = f"{item.class_name}:{item.file_hash}"
        if key in seen_hashes:
            duplicate_counts[item.class_name] += 1
        else:
            seen_hashes[key] = str(item.path)
    return class_counts, duplicate_counts, group_counts


def dedupe_items(items: List[Item]) -> List[Item]:
    chosen: Dict[Tuple[str, str], Item] = {}
    for item in items:
        key = (item.class_name, item.file_hash)
        existing = chosen.get(key)
        if existing is None or str(item.path) < str(existing.path):
            chosen[key] = item
    return sorted(chosen.values(), key=lambda x: (x.class_name, x.group_id, x.path.name))


def split_grouped(items: List[Item], train_ratio: float, seed: int) -> Tuple[List[Item], List[Item]]:
    by_class_group: Dict[str, Dict[str, List[Item]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_class_group[item.class_name][item.group_id].append(item)

    rng = random.Random(seed)
    train_items: List[Item] = []
    val_items: List[Item] = []

    for class_name, groups in sorted(by_class_group.items()):
        group_ids = sorted(groups)
        rng.shuffle(group_ids)
        total_items = sum(len(groups[g]) for g in group_ids)
        target_train = max(1, int(round(total_items * train_ratio)))
        train_count = 0
        train_groups = set()

        for gid in group_ids:
            if train_count >= target_train and train_groups:
                break
            train_groups.add(gid)
            train_count += len(groups[gid])

        for gid, members in groups.items():
            if gid in train_groups:
                train_items.extend(members)
            else:
                val_items.extend(members)

        if not any(item.class_name == class_name for item in val_items):
            # Move one smallest group to val to avoid empty class in validation.
            moved_gid = min(train_groups, key=lambda g: len(groups[g]))
            moved_members = groups[moved_gid]
            train_items = [item for item in train_items if item.group_id != moved_gid]
            val_items.extend(moved_members)

    return train_items, val_items


def copy_split(items: List[Item], dest_root: Path) -> None:
    for item in items:
        class_dir = dest_root / item.class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / item.path.name
        if dest.exists():
            dest = class_dir / f"{item.path.stem}_{item.file_hash[:8]}{item.path.suffix.lower()}"
        shutil.copy2(item.path, dest)


def cross_split_duplicate_count(train_root: Path, val_root: Path) -> int:
    train_hashes = defaultdict(set)
    for path in iter_images(train_root):
        train_hashes[path.parent.name].add(file_hash(path))
    count = 0
    for path in iter_images(val_root):
        if file_hash(path) in train_hashes[path.parent.name]:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backup-suffix", default="leaky_backup")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    backup_root = data_dir / f"{args.backup_suffix}_{args.seed}"
    backup_train = backup_root / "train"
    backup_val = backup_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit("data/train and data/val must exist")

    source_train = train_dir
    source_val = val_dir
    if backup_root.exists():
        source_train = backup_train
        source_val = backup_val
        print(f"Using backup as source: {backup_root}")

    items = collect_items(source_train, source_val)
    class_counts, duplicate_counts, _ = summarize(items)
    deduped = dedupe_items(items)
    train_items, val_items = split_grouped(deduped, args.train_ratio, args.seed)

    print("Current counts by class:")
    for cls in sorted(class_counts):
        print(f"  {cls}: total={class_counts[cls]} duplicate_files={duplicate_counts[cls]}")

    print("\nRebuilt split counts by class:")
    train_counter = Counter(item.class_name for item in train_items)
    val_counter = Counter(item.class_name for item in val_items)
    all_classes = sorted(set(class_counts) | set(train_counter) | set(val_counter))
    for cls in all_classes:
        print(f"  {cls}: train={train_counter[cls]} val={val_counter[cls]}")

    if args.dry_run:
        return

    if not backup_root.exists():
        print(f"\nBacking up current split to {backup_root}")
        shutil.copytree(train_dir, backup_train)
        shutil.copytree(val_dir, backup_val)

    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("Copying clean train split...")
    copy_split(train_items, train_dir)
    print("Copying clean val split...")
    copy_split(val_items, val_dir)

    exact_dup = cross_split_duplicate_count(train_dir, val_dir)
    print(f"\nExact cross-split duplicates after rebuild: {exact_dup}")


if __name__ == "__main__":
    main()
