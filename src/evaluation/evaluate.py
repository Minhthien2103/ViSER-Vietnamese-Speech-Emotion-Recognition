import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from evaluation.plotter import plot_confusion_matrix


EMOTION_LABELS = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Neutral"}


def evaluate(model, X_test, y_test, model_name, source):
    """
        Đánh giá model: in accuracy, classification report, confusion matrix.
        Tự động plot confusion matrix.
    """
    y_pred = model.predict(X_test)

    label_names = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys()) if i in np.unique(y_test)]
    target_ids = sorted(np.unique(y_test))

    print(f"\n{'='*60}")
    print(f"{model_name} ({source})")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, labels = target_ids, target_names = label_names))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels = target_ids))

    # Auto-plot
    plot_confusion_matrix(y_test, y_pred, model_name, source)

    return y_pred
