"""Evaluation primitives, exact confusion matrices and one-vs-rest metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

def compute_metrics(y_true, y_prob, classes):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if p.shape != (len(y), len(classes)):
        raise ValueError('Probability dimensions incompatible')
    if not np.isfinite(p).all() or (p < 0).any() or (not np.allclose(p.sum(axis=1), 1, atol=0.0001)):
        raise ValueError('Probabilities must be nonnegative, finite and sum to one')
    yhat = p.argmax(axis=1)
    cm = confusion_matrix(y, yhat, labels=np.arange(len(classes)))
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    precision = np.divide(tp, pred, out=np.zeros(len(classes)), where=pred > 0)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support > 0)
    f1 = np.divide(2 * tp, support + pred, out=np.zeros(len(classes)), where=support + pred > 0)
    result = {'accuracy': float(tp.sum() / len(y)), 'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall.mean()), 'confusion_matrix': cm, 'per_class_precision': precision, 'per_class_recall': recall, 'per_class_f1': f1}
    if min(support) > 0 and max(support) < len(y):
        au = []
        ap = []
        for i in range(len(classes)):
            binary = (y == i).astype(int)
            au.append(roc_auc_score(binary, p[:, i]))
            ap.append(average_precision_score(binary, p[:, i]))
        result['macro_auroc'] = float(np.mean(au)) if len(classes) > 2 else float(au[0])
        result['macro_auprc'] = float(np.mean(ap)) if len(classes) > 2 else float(ap[0])
    return result
