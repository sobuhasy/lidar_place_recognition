"""Evaluation metrics for place recognition (TODO)."""

def _count_confusion(y_true, y_pred):
    true_iter = [1 if bool(val) else 0 for val in y_true]
    pred_iter = [1 if bool(val) else 0 for val in y_pred]

    tp = sum(1 for t, p in zip(true_iter, pred_iter) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_iter, pred_iter) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(true_iter, pred_iter) if t == 1 and p == 0)
    return tp, fp, tn


def _validate_lengths(y_true, y_scores, y_pred=None):
    if len(y_true) != len(y_scores):
        raise ValueError("Length of y_true and y_scores must be the same")
    if y_pred is not None and len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must be the same")


def precision_recall_f1(y_true, y_pred):
    """Compute precision, recall, and F1-score."""
    _validate_lengths(y_true, y_pred, y_pred=y_pred)
    tp, fp, fn = _count_confusion(y_true, y_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def average_precision(y_true, y_scores):
    """Compute average precision (AP)."""
    _validate_lengths(y_true, y_scores)
    pairs = sorted(zip(y_scores, y_true), key=lambda item: item[0], reverse=True)
    positives = sum(1 for _, label in pairs if bool(label))
    if positives == 0:
        return 0.0
    
    tp = 0
    fp = 0
    precision_sum = 0.0
    for _, label in pairs:
        if bool(label):
            tp += 1
            precision_sum += tp / (tp + fp)
        else:
            fp += 1
    return precision_sum / positives
    


def pr_curve(y_true, y_scores):
    """Compute points for a PR curve."""
    _validate_lengths(y_true, y_scores)
    pairs = sorted(zip(y_scores, y_true), key=lambda item: item[0], reverse=True)
    positives = sum(1 for _, label in pairs if bool(label))
    if positives == 0:
        return [0.0], [0.0], []
    
    precision = []
    recall = []
    thresholds = []
    tp = 0
    fp = 0
    idx = 0
    while idx < len(pairs):
        score = pairs[idx][0]
        while idx < len(pairs) and pairs[idx][0] == score:
            if bool(pairs[idx][1]):
                tp += 1
            else:
                fp += 1
            idx += 1
        precision.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recall.append(tp / positives)
        thresholds.append(score)
    return precision, recall, thresholds

def roc_curve(y_true, y_scores):
    """Compute points for an ROC curve."""
    _validate_lengths(y_true, y_scores)
    pairs = sorted(zip(y_scores, y_true), key=lambda item: item[0], reverse=True)
    positives = sum(1 for _, label in pairs if bool(label))
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return [0, 0], [0, 0], []
    
    fpr = []
    tpr = []
    thresholds = []
    tp = 0
    fp = 0
    idx = 0
    while idx < len(pairs):
        score = pairs[idx][0]
        while idx < len(pairs) and pairs[idx][0] == score:
            if bool(pairs[idx][1]):
                tp += 1
            else:
                fp += 1
            idx += 1
        fpr.append(fp / negatives if negatives else 0.0)
        tpr.append(tp / positives if positives else 0.0)
        thresholds.append(score)
    return fpr, tpr, thresholds
