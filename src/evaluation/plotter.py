import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


RESULTS_DIR = config.BASE_DIR / "results"
RESULTS_DIR.mkdir(parents = True, exist_ok = True)

EMOTION_LABELS = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Neutral"}


def plot_confusion_matrix(y_true, y_pred, model_name, source):
    """
        Vẽ confusion matrix dạng heatmap và lưu file.
    """
    target_ids = sorted(np.unique(y_true))
    label_names = [EMOTION_LABELS[i] for i in target_ids]

    cm = confusion_matrix(y_true, y_pred, labels = target_ids)

    fig, ax = plt.subplots(figsize = (8, 6))
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
                xticklabels = label_names, yticklabels = label_names, ax = ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name} ({source})')

    save_path = RESULTS_DIR / f"cm_{model_name}_{source}.png"
    fig.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)

    print(f"  Plot saved → {save_path}")


def plot_training_history(history, model_name, source):
    """
        Vẽ loss và accuracy curves cho DL training.
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label = 'Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label = 'Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss — {model_name} ({source})')
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label = 'Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label = 'Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Accuracy — {model_name} ({source})')
    axes[1].legend()
    axes[1].grid(True, alpha = 0.3)

    fig.tight_layout()

    save_path = RESULTS_DIR / f"history_{model_name}_{source}.png"
    fig.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)

    print(f"  Plot saved → {save_path}")


def plot_class_distribution(y_train, y_test, source):
    """
        Vẽ phân bố class cho train/test sets.
    """
    target_ids = sorted(np.unique(np.concatenate([y_train, y_test])))
    label_names = [EMOTION_LABELS[i] for i in target_ids]

    train_counts = [np.sum(y_train == i) for i in target_ids]
    test_counts = [np.sum(y_test == i) for i in target_ids]

    x = np.arange(len(label_names))
    width = 0.35

    fig, ax = plt.subplots(figsize = (8, 5))
    ax.bar(x - width / 2, train_counts, width, label = 'Train', color = '#4C72B0')
    ax.bar(x + width / 2, test_counts, width, label = 'Test', color = '#DD8452')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.set_ylabel('Count')
    ax.set_title(f'Class Distribution ({source})')
    ax.legend()
    ax.grid(True, alpha = 0.3, axis = 'y')

    save_path = RESULTS_DIR / f"distribution_{source}.png"
    fig.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)

    print(f"  Plot saved → {save_path}")
