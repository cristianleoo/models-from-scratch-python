import matplotlib.pyplot as plt

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after

    Args:
    -----
        patience (int): Number of epochs to wait before stopping the training.
        verbose (bool): If True, prints a message for each epoch where the loss
                        does not improve.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        """
        Determines if the model should stop training.
        
        Args:
            val_loss (float): The loss of the model on the validation set.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class PlotManager:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 15))

    def plot_losses(self, train_losses, val_losses):
        self.ax[0].plot(train_losses, label='Training Loss')
        self.ax[0].plot(val_losses, label='Validation Loss')
        self.ax[0].set_title('Training and Validation Losses')
        self.ax[0].set_xlabel('Epoch')
        self.ax[0].set_ylabel('Loss')
        self.ax[0].legend()

    # def plot_gradients(self, gradients):
    #     self.ax[1].plot(gradients)
    #     self.ax[1].set_title('Gradients')
    #     self.ax[1].set_xlabel('Step')
    #     self.ax[1].set_ylabel('Gradient')

    def show_plots(self):
        plt.tight_layout()
        plt.show()
