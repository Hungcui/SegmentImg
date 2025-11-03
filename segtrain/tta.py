'''
tta = test time augmentation
'''


import numpy as np, tensorflow as tf

class TTAInference:
    def __init__(self, model, tta_transforms=["flip_h", "flip_v", "rotate_90"]):
        self.model = model
        self.tta_transforms = tta_transforms

    def _apply_transform(self, img, t):
        if t == "flip_h": return np.flip(img, axis=1)
        if t == "flip_v": return np.flip(img, axis=0)
        if t == "rotate_90": return np.rot90(img, k=1, axes=(0,1))
        if t == "rotate_180": return np.rot90(img, k=2, axes=(0,1))
        if t == "rotate_270": return np.rot90(img, k=3, axes=(0,1))
        return img

    def _reverse_transform(self, arr, t):
        if t == "flip_h": return np.flip(arr, axis=1)
        if t == "flip_v": return np.flip(arr, axis=0)
        if t == "rotate_90": return np.rot90(arr, k=-1, axes=(0,1))
        if t == "rotate_180": return np.rot90(arr, k=-2, axes=(0,1))
        if t == "rotate_270": return np.rot90(arr, k=-3, axes=(0,1))
        return arr

    def predict(self, img):
        img_tf = tf.expand_dims(tf.constant(img, dtype=tf.float32), 0)
        outputs = self.model(img_tf, training=False)
        sem_logits = outputs[0] if isinstance(outputs, list) else (outputs.get("sem_logits") if isinstance(outputs, dict) else outputs)
        probs = tf.nn.softmax(sem_logits, axis=-1).numpy()[0]
        preds = [probs]
        for t in self.tta_transforms:
            x = self._apply_transform(img, t)
            out = self.model(tf.expand_dims(tf.constant(x, dtype=tf.float32), 0), training=False)
            sl = out[0] if isinstance(out, list) else (out.get("sem_logits") if isinstance(out, dict) else out)
            p = tf.nn.softmax(sl, axis=-1).numpy()[0]
            preds.append(self._reverse_transform(p, t))
        avg = np.mean(preds, axis=0)
        return np.argmax(avg, axis=-1), avg
