import tensorflow as tf, keras
import numpy as np

def sparse_ce_ignore_index(ignore_index: int, from_logits: bool = True):
    sce = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction='none')
    def loss(y_true, y_pred):
        mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        per_px = sce(y_true_clean, y_pred)
        per_px = tf.where(mask, per_px, tf.zeros_like(per_px))
        denom = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        return tf.reduce_sum(per_px) / denom
    return loss

def weighted_sparse_ce_ignore_index(class_weights: np.ndarray, ignore_index: int, from_logits: bool = True):
    weights_tf = tf.constant(class_weights, dtype=tf.float32)
    sce = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction='none')
    def loss(y_true, y_pred):
        mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        per_px = sce(y_true_clean, y_pred)
        y_true_flat = tf.reshape(y_true_clean, [-1])
        weights_flat = tf.gather(weights_tf, y_true_flat)
        weights_flat = tf.reshape(weights_flat, tf.shape(per_px))
        per_px = per_px * weights_flat
        per_px = tf.where(mask, per_px, tf.zeros_like(per_px))
        denom = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        return tf.reduce_sum(per_px) / denom
    return loss

def focal_loss(alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255, from_logits: bool = True):
    sce = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction='none')
    def loss(y_true, y_pred):
        mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        ce = sce(y_true_clean, y_pred)
        probs = tf.nn.softmax(y_pred, axis=-1) if from_logits else y_pred
        y_true_one_hot = tf.one_hot(tf.cast(y_true_clean, tf.int32), depth=tf.shape(y_pred)[-1])
        p_t = tf.reduce_sum(probs * y_true_one_hot, axis=-1)
        alpha_t = alpha * tf.reduce_sum(y_true_one_hot, axis=-1) + (1 - alpha) * (1 - tf.reduce_sum(y_true_one_hot, axis=-1))
        fl = (alpha_t * tf.pow((1 - p_t), gamma)) * ce
        fl = tf.where(mask, fl, tf.zeros_like(fl))
        denom = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        return tf.reduce_sum(fl) / denom
    return loss

def tversky_loss(alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6, ignore_index: int = 255, from_logits: bool = True):
    def loss(y_true, y_pred):
        mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        y_pred_probs = tf.nn.softmax(y_pred, axis=-1) if from_logits else y_pred
        y_true_one_hot = tf.one_hot(tf.cast(y_true_clean, tf.int32), depth=tf.shape(y_pred)[-1])
        y_true_flat = tf.boolean_mask(tf.reshape(y_true_one_hot, [-1, tf.shape(y_pred)[-1]]), tf.reshape(mask, [-1]))
        y_pred_flat = tf.boolean_mask(tf.reshape(y_pred_probs, [-1, tf.shape(y_pred)[-1]]), tf.reshape(mask, [-1]))
        tp = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat, axis=0)
        fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat), axis=0)
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1.0 - tf.reduce_mean(tversky)
    return loss
