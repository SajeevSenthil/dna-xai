import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from typing import Dict, Any, Union

def compute_metrics(
    y_true: Union[np.ndarray, list], 
    y_pred: Union[np.ndarray, list], 
    y_prob: Union[np.ndarray, list]
) -> Dict[str, float]:
    """
    Computes standard classification evaluation metrics.
    Supports binary classification.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        # Falls back if there is only 1 class in the batch/split
        auroc = 0.5
        
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc)
    }

def plot_confusion_matrix(
    y_true: Union[np.ndarray, list], 
    y_pred: Union[np.ndarray, list], 
    save_path: str,
    title: str = "Confusion Matrix"
) -> None:
    """
    Generates and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"]
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.ylabel("True Label", fontsize=11, labelpad=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_pr_curves(
    y_true: Union[np.ndarray, list], 
    y_prob: Union[np.ndarray, list], 
    save_dir: str,
    prefix: str = ""
) -> None:
    """
    Generates and saves ROC and Precision-Recall curves.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    os.makedirs(save_dir, exist_ok=True)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.5
        
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auroc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11, labelpad=10)
    plt.ylabel("True Positive Rate", fontsize=11, labelpad=10)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}roc_curve.png"), dpi=300)
    plt.close()

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="forestgreen", lw=2, label=f"PR Curve (AP = {ap_score:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11, labelpad=10)
    plt.ylabel("Precision", fontsize=11, labelpad=10)
    plt.title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}pr_curve.png"), dpi=300)
    plt.close()
