import numpy as np

def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int=255):
    mask = target != ignore_index
    pred = pred[mask]; target = target[mask]
    k = (target.astype(np.int64) * num_classes + pred.astype(np.int64))
    bincount = np.bincount(k, minlength=num_classes**2)
    return bincount.reshape(num_classes, num_classes)

def miou_from_confmat(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=1).astype(np.float64) - tp
    fn = cm.sum(axis=0).astype(np.float64) - tp
    denom = tp + fp + fn + 1e-6
    iou = (tp / denom)
    mean_iou = float(np.mean(iou))
    return mean_iou, list(map(float, iou))
