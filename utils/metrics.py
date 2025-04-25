
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

def compute_metrics(y_true, y_pred):
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, average='weighted'),
        precision=precision_score(y_true, y_pred, average='weighted'),
        recall=recall_score(y_true, y_pred, average='weighted')
    )

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    if save_path: plt.savefig(save_path)
