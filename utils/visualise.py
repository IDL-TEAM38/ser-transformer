
import matplotlib.pyplot as plt

def plot_training(history, save_path=None):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train')
    if history['val_loss'][0] is not None:
        plt.plot(epochs, history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(); plt.legend()
    plt.subplot(1,2,2)
    if history['val_acc'][0] is not None:
        plt.plot(epochs, history['val_acc'], label='Val Acc'); plt.grid(); plt.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
