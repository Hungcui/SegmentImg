import numpy as np, keras, tensorflow as tf
from pathlib import Path
from segtrain.metrics import compute_confusion_matrix, miou_from_confmat

class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, num_classes: int, ignore_index: int, ckpt_path: Path):
        super().__init__()
        self.val_ds = val_ds
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.best_miou = 0.0
        self.ckpt_path = ckpt_path

    def on_epoch_end(self, epoch, logs=None):
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        correct_px, total_px = 0, 0
        total_bce, n_samples = 0.0, 0
        bce = keras.losses.BinaryCrossentropy(from_logits=True)

        for imgs, y in self.val_ds:
            masks = y["sem_logits"].numpy()
            boundary_t = y["boundary_logits"].numpy()
            outputs = self.model(imgs, training=False)
            if isinstance(outputs, list):
                sem_logits = outputs[0] if len(outputs) else None
                boundary_logits = outputs[-1] if len(outputs) > 1 else None
            else:
                sem_logits = outputs.get("sem_logits") if isinstance(outputs, dict) else outputs
                boundary_logits = outputs.get("boundary_logits") if isinstance(outputs, dict) else None

            if sem_logits is not None:
                preds = tf.argmax(sem_logits, axis=-1).numpy().astype(np.int64)
                cm += compute_confusion_matrix(preds, masks, self.num_classes, self.ignore_index)
                valid = (masks != self.ignore_index)
                correct_px += (preds == masks)[valid].sum()
                total_px += valid.sum()
            if boundary_logits is not None and boundary_t is not None:
                total_bce += bce(boundary_t, boundary_logits).numpy() * imgs.shape[0]
            n_samples += imgs.shape[0]

        pixacc = correct_px / max(1, total_px)
        miou, class_ious = miou_from_confmat(cm)
        val_bce = total_bce / max(1, n_samples)
        print(f"[Eval] valPA={pixacc:.3f}  valmIoU={miou:.3f}  valBCE={val_bce:.4f}  per-class IoU={np.round(class_ious,3)}")

        if miou > self.best_miou:
            self.best_miou = miou
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.ckpt_path.as_posix())
            print(f"Saved best to {self.ckpt_path} (mIoU {miou:.3f})")
