# inference.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from rembg import remove

class PlantDiseaseClassifier:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.classes = [
            'citrus_black_spot', 'citrus_canker', 'citrus_foliage_damage',
            'citrus_greening', 'citrus_healthy', 'citrus_mealybugs',
            'citrus_melanose', 'mango_anthracnose', 'mango_bacterial_canker',
            'mango_cutting_weevil', 'mango_die_back', 'mango_gall_midge',
            'mango_healthy', 'mango_powdery_mildew', 'mango_sooty_mould'
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.502275, 0.528588, 0.474627],
                               std=[0.251866, 0.251658, 0.335502])
        ])
        
        # Load model (TorchScript)
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    @staticmethod
    def _segment_image(image: Image.Image) -> Image.Image:
        image = image.convert('RGB')
        try:
            segmented = remove(image)
            if segmented.mode != 'RGBA':
                return segmented.convert('RGB')

            canvas = Image.new('RGB', segmented.size, (0, 0, 0))
            canvas.paste(segmented, mask=segmented.getchannel('A'))
            return canvas
        except Exception:
            return image
    
    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = self._segment_image(img)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
        
        conf, pred_idx = torch.max(probs, dim=0)
        
        return {
            'class': self.classes[pred_idx.item()],
            'confidence': float(conf.item()),
            'all_probabilities': {
                cls: float(prob) 
                for cls, prob in zip(self.classes, probs.cpu())
            }
        }

classifier = PlantDiseaseClassifier(r'models/mobilenet_v2_plant_disease.pt')
result = classifier.predict(r"PATH") # replace with the path of intended image
print(result)
