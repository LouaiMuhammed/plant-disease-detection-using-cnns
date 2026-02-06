"""
Early stopping utility to prevent overfitting
"""


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving
    """
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return False