import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score

from .plot import plot_confusion_matrix


def compute_metrics(df, class_names=None):
    """Computes metrics for the given dataframe."""
    acc = accuracy_score(df["true"], df["pred"])
    prec = precision_score(df["true"], df["pred"], average="weighted")
    rec = recall_score(df["true"], df["pred"], average="weighted")
    cm = confusion_matrix(df["true"], df["pred"])
    if class_names is None:
        class_names = np.arange(cm.shape[0]).astype(str)

    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", rec)

    # class_names = [f'{i}' for i in sorted(df['pred'].unique())]
    print("Class. report:\n", classification_report(df["true"], df["pred"], target_names=class_names))
    plot_confusion_matrix(cm, class_names)

    print("Conf. matrix:\n", cm)
    metrics = {"accuracy": acc, "precision_w": prec, "recall_w": rec, "cm": cm.tolist()}
    return metrics
