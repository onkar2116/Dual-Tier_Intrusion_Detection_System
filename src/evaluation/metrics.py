import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


def compute_all_metrics(y_true, y_pred, y_probs=None, average='weighted'):
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (optional, for AUC-ROC)
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')

    Returns:
        dict of all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    # False positive/negative rates (binary)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        # Multi-class: compute per-class and average
        n_classes = cm.shape[0]
        fprs = []
        fnrs = []
        for i in range(n_classes):
            tp_i = cm[i, i]
            fn_i = cm[i, :].sum() - tp_i
            fp_i = cm[:, i].sum() - tp_i
            tn_i = cm.sum() - tp_i - fn_i - fp_i
            fprs.append(fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0)
            fnrs.append(fn_i / (fn_i + tp_i) if (fn_i + tp_i) > 0 else 0)
        metrics['fpr'] = float(np.mean(fprs))
        metrics['fnr'] = float(np.mean(fnrs))

    # AUC-ROC
    if y_probs is not None:
        try:
            if len(np.unique(y_true)) == 2:
                probs = y_probs[:, 1] if y_probs.ndim > 1 else y_probs
                metrics['auc_roc'] = roc_auc_score(y_true, probs)
            else:
                metrics['auc_roc'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average=average
                )
        except ValueError:
            metrics['auc_roc'] = 0.0

    return metrics


def compute_robust_accuracy(y_true, y_pred_clean, y_pred_adv):
    """
    Compute clean and robust accuracy.

    Args:
        y_true: True labels
        y_pred_clean: Predictions on clean data
        y_pred_adv: Predictions on adversarial data
    """
    clean_acc = accuracy_score(y_true, y_pred_clean)
    robust_acc = accuracy_score(y_true, y_pred_adv)
    accuracy_drop = clean_acc - robust_acc

    return {
        'clean_accuracy': clean_acc,
        'robust_accuracy': robust_acc,
        'accuracy_drop': accuracy_drop,
        'robustness_ratio': robust_acc / clean_acc if clean_acc > 0 else 0
    }
