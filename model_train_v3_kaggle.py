# Improved Model Training Script with Advanced Features
# Based on model_train_v2.py with enhancements:
# - Advanced data augmentation
# - Class imbalance handling (Focal Loss, Tversky Loss)
# - Attention U-Net architecture
# - EfficientNet backbone
# - Multiple loss functions (CE, Focal, Tversky)
# - Instance Segmentation (from semantic + boundary)

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys
import random
import keras
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

import tensorflow as tf
from keras import layers, Model, ops
try:
    from keras import mixed_precision
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    MIXED_PRECISION_AVAILABLE = False

# Set mixed precision policy to float32 IMMEDIATELY after import to prevent dtype conflicts
# This must be done before any model building operations
if MIXED_PRECISION_AVAILABLE:
    try:
        mixed_precision.set_global_policy('float32')
    except:
        pass
else:
    tf.keras.backend.set_floatx('float32')

# Also disable mixed precision in TensorFlow config (for Kaggle/Colab environments)
try:
    tf.config.experimental.enable_mixed_precision_graph_rewrite(False)
except:
    pass

from skimage.segmentation import find_boundaries
from scipy import ndimage as ndi
from scipy.ndimage import binary_closing, binary_opening
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Try to import EfficientNet backbones
try:
    from keras.applications import EfficientNetB0, EfficientNetB3, EfficientNetB4
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("Warning: EfficientNet not available. Using standard encoder.")


'''
Understanding data
    labelmap: each animal have specific color
    segmentation object: color: red, just for detect object, NOT classification
        Inside file: segmented image
    segmentation class: each animal has specific color
        Inside file: segmented image 
    ImageSets: file text include name of original image used to train
'''

def read_labelmap(labelmap_path: Path): 
    """
    Reads a labelmap file, ignoring blank lines and lines starting with '#'.
    Returns two lists: names (labels) and colors (RGB tuples).
    """
    if not labelmap_path.exists():
        raise FileNotFoundError(f"File not found: {labelmap_path}")

    names, colors = [], []
    text = Path(labelmap_path).read_text(encoding="utf-8").splitlines()

    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Missing colon in line: {line}")

        name, rest = line.split(":", 1)
        name = name.strip()
        color_field = rest.split(":", 1)[0]
        comps = color_field.split(",")

        if len(comps) != 3:
            raise ValueError(f"RGB must have 3 components: {line}")

        try:
            r, g, b = [int(c.strip()) for c in comps]
        except Exception as e:
            raise ValueError(f"Non-integer RGB values in line: {line}") from e

        names.append(name)
        colors.append((r, g, b))

    return names, colors

def build_color_to_index(colors: List[Tuple[int,int,int]]) -> Dict[Tuple[int,int,int], int]:
    return {tuple(map(int, c)): i for i, c in enumerate(colors)}

def mask_rgb_to_index(mask_img: Image.Image, color_to_index: Dict[Tuple[int,int,int], int], ignore_index=255) -> np.ndarray:
    """
    Convert an RGB palette/truecolor mask (H,W,3) into class indices (H,W).
    Any pixel color not found in color_to_index becomes ignore_index.
    """
    m = np.array(mask_img.convert("RGB"), dtype=np.uint8)
    h, w, _ = m.shape
    flat = m.reshape(-1, 3)
    out = np.full((h*w,), ignore_index, dtype=np.uint8)
    keys = (flat[:,0].astype(np.int32) << 16) | (flat[:,1].astype(np.int32) << 8) | flat[:,2].astype(np.int32)
    lut = {}
    for (r,g,b), idx in color_to_index.items():
        lut[(r<<16) | (g<<8) | b] = idx
    for k, idx in lut.items():
        out[keys == k] = idx
    return out.reshape(h, w)

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ========== METRICS ==========
def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int=255):
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    k = (target.astype(np.int64) * num_classes + pred.astype(np.int64))
    bincount = np.bincount(k, minlength=num_classes**2)
    return bincount.reshape(num_classes, num_classes)

def miou_from_confmat(cm: np.ndarray) -> Tuple[float, List[float]]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=1).astype(np.float64) - tp
    fn = cm.sum(axis=0).astype(np.float64) - tp
    denom = tp + fp + fn + 1e-6
    iou = (tp / denom)
    mean_iou = float(np.mean(iou))
    return mean_iou, list(map(float, iou))

# ========== ADVANCED DATA AUGMENTATION ==========
class AdvancedAugmentation:
    """Advanced augmentation with rotation, color jitter, elastic transform"""
    def __init__(self, 
                 rotation_range=30,
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 saturation_range=(0.8, 1.2),
                 elastic_alpha=100,
                 elastic_sigma=10,
                 elastic_prob=0.5):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob

    def _elastic_transform(self, image, mask):
        """Elastic deformation"""
        if np.random.rand() > self.elastic_prob:
            return image, mask
        
        shape = image.shape[:2]
        dx = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        dy = ndi.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.elastic_sigma) * self.elastic_alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply to image
        if len(image.shape) == 3:
            distorted_image = np.zeros_like(image)
            for i in range(image.shape[2]):
                distorted_image[:, :, i] = ndi.map_coordinates(image[:, :, i], indices, order=1, mode='reflect').reshape(shape)
        else:
            distorted_image = ndi.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        
        # Apply to mask with nearest neighbor
        distorted_mask = ndi.map_coordinates(mask.astype(np.float32), indices, order=0, mode='reflect').reshape(shape).astype(mask.dtype)
        
        return distorted_image, distorted_mask

    def _color_jitter(self, image):
        """Color jitter: brightness, contrast, saturation"""
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        # Brightness
        if self.brightness_range:
            enhancer = ImageEnhance.Brightness(img_pil)
            factor = np.random.uniform(*self.brightness_range)
            img_pil = enhancer.enhance(factor)
        
        # Contrast
        if self.contrast_range:
            enhancer = ImageEnhance.Contrast(img_pil)
            factor = np.random.uniform(*self.contrast_range)
            img_pil = enhancer.enhance(factor)
        
        # Saturation
        if self.saturation_range:
            enhancer = ImageEnhance.Color(img_pil)
            factor = np.random.uniform(*self.saturation_range)
            img_pil = enhancer.enhance(factor)
        
        return np.array(img_pil).astype(np.float32) / 255.0

    def _random_rotation(self, image, mask):
        """Random rotation"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine((image * 255).astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR)
        rotated_mask = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST)
        
        return rotated_img.astype(np.float32) / 255.0, rotated_mask.astype(mask.dtype)

    def apply(self, image, mask):
        """Apply all augmentations"""
        # Rotation
        if self.rotation_range > 0 and np.random.rand() < 0.5:
            image, mask = self._random_rotation(image, mask)
        
        # Color jitter
        if np.random.rand() < 0.5:
            image = self._color_jitter(image)
        
        # Elastic transform
        image, mask = self._elastic_transform(image, mask)
        
        return image, mask

# ========== LOSS FUNCTIONS ==========
def sparse_ce_ignore_index(ignore_index: int, from_logits: bool = True):
    """
    SparseCategoricalCrossentropy that masks out ignore_index.
    y_true: (B,H,W) int
    y_pred: (B,H,W,C) logits
    """
    sce = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction='none')
    def loss(y_true, y_pred):
        mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        per_px = sce(y_true_clean, y_pred)
        per_px = tf.where(mask, per_px, tf.zeros_like(per_px))
        denom = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        return tf.reduce_sum(per_px) / denom
    return loss

def focal_loss(alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255, from_logits: bool = True):
    """Focal Loss for addressing class imbalance"""
    sce = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction='none')
    
    def loss(y_true, y_pred):
        mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        ce = sce(y_true_clean, y_pred)
        
        # Compute probabilities
        if from_logits:
            probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            probs = y_pred
        
        # Get probability of true class
        y_true_one_hot = tf.one_hot(tf.cast(y_true_clean, tf.int32), depth=tf.shape(y_pred)[-1])
        p_t = tf.reduce_sum(probs * y_true_one_hot, axis=-1)
        
        # Focal term
        alpha_t = alpha * tf.reduce_sum(y_true_one_hot, axis=-1) + (1 - alpha) * (1 - tf.reduce_sum(y_true_one_hot, axis=-1))
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        focal_loss = focal_weight * ce
        focal_loss = tf.where(mask, focal_loss, tf.zeros_like(focal_loss))
        denom = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        return tf.reduce_sum(focal_loss) / denom
    return loss

def tversky_loss(alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6, ignore_index: int = 255, from_logits: bool = True):
    """Tversky Loss - generalization of Dice loss"""
    def loss(y_true, y_pred):
        mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        
        if from_logits:
            y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred_probs = y_pred
        
        y_true_one_hot = tf.one_hot(tf.cast(y_true_clean, tf.int32), depth=tf.shape(y_pred)[-1])
        
        # Flatten
        y_true_flat = tf.reshape(y_true_one_hot, [-1, tf.shape(y_pred)[-1]])
        y_pred_flat = tf.reshape(y_pred_probs, [-1, tf.shape(y_pred)[-1]])
        mask_flat = tf.reshape(mask, [-1])
        
        # Apply mask
        y_true_flat = tf.boolean_mask(y_true_flat, mask_flat)
        y_pred_flat = tf.boolean_mask(y_pred_flat, mask_flat)
        
        # Tversky index per class
        tp = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
        fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat, axis=0)
        fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat), axis=0)
        
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        return 1.0 - tf.reduce_mean(tversky)
    return loss

# ========== MODEL ARCHITECTURES ==========
def double_conv_block(x, n_filters, use_bn=True):
    """Double convolution block with float32 enforcement"""
    # Ensure input is float32
    x = tf.cast(x, tf.float32)
    x = layers.Conv2D(n_filters, 3, padding="same", kernel_initializer="he_normal", use_bias=not use_bn)(x)
    x = tf.cast(x, tf.float32)
    if use_bn: 
        x = layers.BatchNormalization()(x)
        x = tf.cast(x, tf.float32)
    x = layers.ReLU()(x)
    x = tf.cast(x, tf.float32)
    x = layers.Conv2D(n_filters, 3, padding="same", kernel_initializer="he_normal", use_bias=not use_bn)(x)
    x = tf.cast(x, tf.float32)
    if use_bn: 
        x = layers.BatchNormalization()(x)
        x = tf.cast(x, tf.float32)
    x = layers.ReLU()(x)
    x = tf.cast(x, tf.float32)
    return x

def attention_gate(g, x, n_filters):
    """Attention gate for Attention U-Net"""
    theta_g = layers.Conv2D(n_filters, 1, padding="same")(g)
    phi_x = layers.Conv2D(n_filters, 1, padding="same")(x)
    
    add = layers.Add()([theta_g, phi_x])
    relu = layers.ReLU()(add)
    psi = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(relu)
    
    return layers.Multiply()([x, psi])

def downsample_block(x, n_filters, dropout=0.2, use_bn=True):
    f = double_conv_block(x, n_filters, use_bn=use_bn)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout)(p)
    return f, p

def upsample_block(x, conv_feature, n_filters, dropout=0.2, use_bn=True, use_attention=False):
    x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding="same")(x)
    if use_attention:
        conv_feature = attention_gate(x, conv_feature, n_filters)
    x = layers.Concatenate(axis=-1)([x, conv_feature])
    x = layers.Dropout(dropout)(x)
    x = double_conv_block(x, n_filters, use_bn=use_bn)
    return x

def build_attention_unet(input_shape=(512, 512, 3), num_classes=6, dropout=0.2, use_batchnorm=True):
    """Attention U-Net architecture"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    f1, p1 = downsample_block(inputs, 64, dropout, use_batchnorm)
    f2, p2 = downsample_block(p1, 128, dropout, use_batchnorm)
    f3, p3 = downsample_block(p2, 256, dropout, use_batchnorm)
    f4, p4 = downsample_block(p3, 512, dropout, use_batchnorm)
    
    # Bottleneck
    bottleneck = double_conv_block(p4, 1024, use_bn=use_batchnorm)
    
    # Decoder with attention
    u6 = upsample_block(bottleneck, f4, 512, dropout, use_batchnorm, use_attention=True)
    u7 = upsample_block(u6, f3, 256, dropout, use_batchnorm, use_attention=True)
    u8 = upsample_block(u7, f2, 128, dropout, use_batchnorm, use_attention=True)
    u9 = upsample_block(u8, f1, 64, dropout, use_batchnorm, use_attention=True)
    
    sem_logits = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(u9)
    boundary_logits = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(u9)
    
    model = Model(inputs, [sem_logits, boundary_logits], name="AttentionUNet")
    return model

def build_unet_with_backbone(input_shape=(512, 512, 3), num_classes=6, backbone="efficientnet", backbone_name="EfficientNetB0", dropout=0.2):
    """U-Net with pretrained backbone encoder - Fixed to use float32 to avoid mixed precision issues"""
    # Ensure float32 policy is set (should already be set at module level, but double-check)
    if MIXED_PRECISION_AVAILABLE:
        try:
            current_policy = str(mixed_precision.global_policy())
            if 'float32' not in current_policy.lower():
                mixed_precision.set_global_policy('float32')
                print(f"⚠️  Warning: Mixed precision policy was {current_policy}, changed to float32")
        except:
            pass
    
    # Force float32 dtype for input layer
    inputs = layers.Input(shape=input_shape, dtype='float32')
    
    # Get backbone - policy should already be float32 from module level
    if backbone == "efficientnet" and EFFICIENTNET_AVAILABLE:
        # Load EfficientNet with float32 policy enforced
        if backbone_name == "EfficientNetB0":
            backbone_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
        elif backbone_name == "EfficientNetB3":
            backbone_model = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
        elif backbone_name == "EfficientNetB4":
            backbone_model = EfficientNetB4(include_top=False, weights="imagenet", input_tensor=inputs)
        else:
            backbone_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
        
        # CRITICAL: Convert all backbone weights to float32 immediately after loading
        print("🔄 Converting EfficientNet backbone weights to float32...")
        for layer in backbone_model.layers:
            if hasattr(layer, 'weights') and layer.weights:
                for weight in layer.weights:
                    if weight.dtype != tf.float32:
                        # Get current weight value and convert to float32
                        weight_value = tf.cast(weight.value(), tf.float32)
                        weight.assign(weight_value)
        print("✅ Backbone weights converted to float32")
        
        # Get skip connections from EfficientNet
        # EfficientNet structure: block1, block2, block3, block4, block5, block6, block7
        skip_layers = []
        for block_name in ["block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation"]:
            try:
                skip_output = backbone_model.get_layer(block_name).output
                # CRITICAL: Force skip connections to float32
                skip_output = tf.cast(skip_output, tf.float32)
                skip_layers.append(skip_output)
            except:
                pass
        # CRITICAL: Force encoder output to float32
        encoder_output = tf.cast(backbone_model.output, tf.float32)
    else:
        # Fallback to standard encoder
        return build_unet_with_boundary(input_shape, num_classes, dropout)
    
    # Decoder - ensure all operations use float32
    x = encoder_output
    # Ensure x is float32
    x = tf.cast(x, tf.float32)
    
    for i, skip in enumerate(reversed(skip_layers)):
        n_filters = 512 // (2 ** i)
        # Ensure skip connection is float32
        skip = tf.cast(skip, tf.float32)
        x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding="same")(x)
        x = tf.cast(x, tf.float32)  # Force float32 after transpose
        x = layers.Concatenate(axis=-1)([x, skip])
        x = tf.cast(x, tf.float32)  # Force float32 after concat
        x = double_conv_block(x, n_filters, use_bn=True)
        x = tf.cast(x, tf.float32)  # Force float32 after conv block
        x = layers.Dropout(dropout)(x)
        x = tf.cast(x, tf.float32)  # Force float32 after dropout
    
    # Final upsampling
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
    x = tf.cast(x, tf.float32)
    x = double_conv_block(x, 64, use_bn=True)
    x = tf.cast(x, tf.float32)
    
    sem_logits = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(x)
    boundary_logits = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(x)
    
    # CRITICAL: Force final outputs to float32
    sem_logits = tf.cast(sem_logits, tf.float32)
    boundary_logits = tf.cast(boundary_logits, tf.float32)
    
    model = Model(inputs, [sem_logits, boundary_logits], name=f"UNet_{backbone_name}")
    return model

def build_unet_with_boundary(input_shape=(512, 512, 3), num_classes=6, dropout=0.2, use_batchnorm=True):
    """Standard U-Net with boundary head"""
    inputs = layers.Input(shape=input_shape)
    f1, p1 = downsample_block(inputs, 64, dropout, use_batchnorm)
    f2, p2 = downsample_block(p1, 128, dropout, use_batchnorm)
    f3, p3 = downsample_block(p2, 256, dropout, use_batchnorm)
    f4, p4 = downsample_block(p3, 512, dropout, use_batchnorm)
    bottleneck = double_conv_block(p4, 1024, use_bn=use_batchnorm)
    u6 = upsample_block(bottleneck, f4, 512, dropout, use_batchnorm)
    u7 = upsample_block(u6, f3, 256, dropout, use_batchnorm)
    u8 = upsample_block(u7, f2, 128, dropout, use_batchnorm)
    u9 = upsample_block(u8, f1, 64, dropout, use_batchnorm)
    sem_logits = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(u9)
    boundary_logits = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(u9)
    model = Model(inputs, [sem_logits, boundary_logits], name="UNetBoundary")
    return model

def make_boundary_targets(mask_batch: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    out = []
    for m in mask_batch:
        mm = m.copy()
        mm[mm == ignore_index] = 0
        b = find_boundaries(mm, mode="inner").astype(np.float32)
        out.append(b[..., None])
    return np.stack(out, axis=0)

def instances_from_sem_and_boundary(sem_logits, boundary_logits, thing_class_ids: List[int],
                                    sem_thresh: float = 0.5, boundary_thresh: float = 0.5) -> np.ndarray:
    if isinstance(sem_logits, tf.Tensor): sem_logits = sem_logits.numpy()
    if isinstance(boundary_logits, tf.Tensor): boundary_logits = boundary_logits.numpy()
    
    if sem_logits.ndim == 4:
        if sem_logits.shape[0] == 1 and sem_logits.shape[1] != 1:
            sem_prob = tf.nn.softmax(sem_logits, axis=1).numpy()[0]
        else:
            sem_prob = tf.nn.softmax(np.transpose(sem_logits, (0,3,1,2)), axis=1).numpy()[0]
    else:
        sem_prob = tf.nn.softmax(np.transpose(sem_logits, (2,0,1)), axis=0).numpy()
    
    if boundary_logits.ndim == 4:
        if boundary_logits.shape[-1] == 1:
            boundary_prob = tf.nn.sigmoid(boundary_logits)[0,...,0].numpy()
        else:
            boundary_prob = tf.nn.sigmoid(boundary_logits)[0,0].numpy()
    else:
        boundary_prob = tf.nn.sigmoid(boundary_logits[...,0]).numpy()
    
    H, W = boundary_prob.shape
    instance_map = np.zeros((H, W), dtype=np.int32)
    cur_label = 0
    
    for cid in thing_class_ids:
        fg = sem_prob[cid]
        fg_bin = fg >= sem_thresh
        mask = fg_bin & (boundary_prob < boundary_thresh)
        if mask.sum() == 0:
            continue

        # Morphological operations to smooth the mask and reduce noise
        # This helps create larger, more coherent instances
        selem_size = max(3, int(np.sqrt(mask.sum()) / 50))  # Adaptive size
        selem = np.ones((selem_size, selem_size), dtype=bool)
        mask = binary_opening(mask, structure=selem)  # Remove small noise
        mask = binary_closing(mask, structure=selem)  # Fill small holes
        
        if mask.sum() == 0:
            continue
        
        distance = ndi.distance_transform_edt(mask)
        
        # Use larger footprint and min_distance to reduce number of markers
        # This creates fewer, larger instances (each animal = 1 instance ideally)
        min_distance = max(10, int(np.sqrt(mask.sum()) / 10))  # Larger min_distance
        coords = peak_local_max(distance, 
                               footprint=np.ones((7,7)),  # Even larger footprint
                               min_distance=min_distance,  # Minimum distance between peaks
                               labels=mask,
                               threshold_abs=distance.max() * 0.4)  # Higher threshold for significant peaks
        
        markers = np.zeros_like(distance, dtype=np.int32)
        for i, (r,c) in enumerate(coords, start=1):
            markers[r, c] = i

        if markers.max() == 0:
            # Fallback: use connected components
            markers, _ = ndi.label(mask)

        labels_ = watershed(-distance, markers, mask=mask).astype(np.int32)
        labels_[labels_>0] += cur_label
        instance_map[labels_>0] = labels_[labels_>0]
        cur_label = instance_map.max()
    
    return instance_map

# ========== ENHANCED DATASET ==========
class EnhancedMultiRootVOCDataset:
    """Enhanced dataset with advanced augmentation and class weight computation"""
    def __init__(self, roots: List[str], image_set: str,
                 names: List[str], colors: List[Tuple[int,int,int]],
                 crop_size: int = 512, random_scale=(0.5, 2.0),
                 hflip_prob: float = 0.5, ignore_index: int = 255,
                 use_advanced_aug: bool = True):
        self.roots = [Path(r) for r in roots]
        self.image_set = image_set
        self.names, self.colors = names, colors
        self.ignore_index = ignore_index
        self.crop_size, self.random_scale, self.hflip_prob = crop_size, random_scale, hflip_prob
        self.color_to_index = build_color_to_index(colors)
        self.use_advanced_aug = use_advanced_aug and (image_set == "train")
        
        if self.use_advanced_aug:
            self.aug = AdvancedAugmentation()
        
        self.samples = []
        for root in self.roots:
            set_file = root / "ImageSets" / "Segmentation" / f"{image_set}.txt"
            ids = [s.strip() for s in set_file.read_text().splitlines() if s.strip()]
            for img_id in ids:
                self.samples.append((root, img_id))
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __len__(self): 
        return len(self.samples)
    
    def _load_sample(self, root: Path, img_id: str):
        img_dir, mask_dir = root / "JPEGImages", root / "SegmentationClass"
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            alt = img_dir / f"{img_id}.png"
            img_path = alt if alt.exists() else img_path
        mask_path = mask_dir / f"{img_id}.png"
        
        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path)
        mask = mask_rgb_to_index(mask_rgb, self.color_to_index, ignore_index=self.ignore_index)
        return image, mask
    
    def _random_resize(self, img, mask):
        if self.random_scale:
            s = np.random.uniform(*self.random_scale)
            new_w, new_h = int(img.width * s), int(img.height * s)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            mask = Image.fromarray(mask, mode="L").resize((new_w, new_h), Image.NEAREST)
            mask = np.array(mask, dtype=np.int64)
        return img, mask
    
    def _random_crop(self, img, mask):
        th, tw = self.crop_size, self.crop_size
        if img.height < th or img.width < tw:
            pad_h, pad_w = max(0, th - img.height), max(0, tw - img.width)
            img = Image.fromarray(np.pad(np.array(img), ((0,pad_h),(0,pad_w),(0,0)),
                                         mode="constant", constant_values=0).astype(np.uint8))
            mask = np.pad(mask, ((0,pad_h),(0,pad_w)), mode="constant", constant_values=self.ignore_index)
        
        i = np.random.randint(0, img.height - th + 1)
        j = np.random.randint(0, img.width - tw + 1)
        img = img.crop((j, i, j+tw, i+th))
        mask = mask[i:i+th, j:j+tw]
        return img, mask
    
    def _hflip(self, img, mask):
        if np.random.rand() < self.hflip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask[:, ::-1]
        return img, mask
    
    def _center_crop_or_resize(self, img, mask):
        short = min(img.width, img.height)
        if short < self.crop_size:
            s = self.crop_size / short
            img = img.resize((int(img.width*s), int(img.height*s)), Image.BILINEAR)
            mask = Image.fromarray(mask, mode="L").resize((int(mask.shape[1]*s), int(mask.shape[0]*s)), Image.NEAREST)
            mask = np.array(mask, dtype=np.int64)
        th, tw = self.crop_size, self.crop_size
        i = max(0, (img.height - th)//2)
        j = max(0, (img.width - tw)//2)
        img = img.crop((j, i, j+tw, i+th))
        mask = mask[i:i+th, j:j+tw]
        return img, mask
    
    def get_item(self, idx):
        root, img_id = self.samples[idx]
        img, mask = self._load_sample(root, img_id)
        
        if self.image_set == "train":
            img, mask = self._random_resize(img, mask)
            img, mask = self._random_crop(img, mask)
            img, mask = self._hflip(img, mask)
            
            # Convert to numpy for advanced augmentation
            img_np = np.asarray(img, dtype=np.float32) / 255.0
            if self.use_advanced_aug:
                img_np, mask = self.aug.apply(img_np, mask)
        else:
            img, mask = self._center_crop_or_resize(img, mask)
            img_np = np.asarray(img, dtype=np.float32) / 255.0
        
        # Normalize
        img_np = (img_np - self.mean) / self.std
        mask_np = mask.astype(np.int64)
        return img_np, mask_np

# ========== TF DATA PIPELINE ==========
def make_tf_dataset(voc: EnhancedMultiRootVOCDataset, batch_size: int, shuffle: bool, ignore_index: int):
    indices = np.arange(len(voc), dtype=np.int32)
    
    def _py_load(idx):
        img, mask = voc.get_item(int(idx))
        bt = make_boundary_targets(np.expand_dims(mask, 0), ignore_index=ignore_index)[0]
        return img.astype(np.float32), mask.astype(np.int32), bt.astype(np.float32)
    
    def _tf_map(idx):
        img, mask, bt = tf.numpy_function(_py_load, [idx], [tf.float32, tf.int32, tf.float32])
        img.set_shape([None, None, 3])
        mask.set_shape([None, None])
        bt.set_shape([None, None, 1])
        return img, {"sem_logits": mask, "boundary_logits": bt}
    
    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(voc), reshuffle_each_iteration=True)
    ds = ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ========== EVALUATION CALLBACK ==========
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
        total_bce = 0.0
        n_samples = 0
        
        for batch in self.val_ds:
            imgs, y = batch
            masks = y["sem_logits"].numpy()
            boundary_t = y["boundary_logits"].numpy()
            
            outputs = self.model(imgs, training=False)
            if isinstance(outputs, list):
                sem_logits = outputs[0] if len(outputs) > 0 else None
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
                bce = keras.losses.BinaryCrossentropy(from_logits=True)
                total_bce += bce(boundary_t, boundary_logits).numpy() * imgs.shape[0]
            
            n_samples += imgs.shape[0]
        
        pixacc = correct_px / max(1, total_px)
        miou, class_ious = miou_from_confmat(cm)
        val_bce = total_bce / max(1, n_samples)
        
        # Tính val_loss tổng hợp từ semantic và boundary losses
        # Sử dụng negative mIoU như loss để ReduceLROnPlateau có thể monitor (mode='min')
        val_loss = -miou  # Negative mIoU: càng thấp càng tốt (giống loss)
        
        # Thêm metrics vào logs để các callbacks khác có thể monitor
        if logs is not None:
            logs['val_loss'] = val_loss
            logs['val_miou'] = miou
            logs['val_pa'] = pixacc
            logs['val_bce'] = val_bce
        
        print(f"[Eval] valPA={pixacc:.3f}  valmIoU={miou:.3f}  valBCE={val_bce:.4f}  per-class IoU={np.round(class_ious,3)}")
        
        if miou > self.best_miou:
            self.best_miou = miou
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.ckpt_path.as_posix())
            print(f"Saved best to {self.ckpt_path} (mIoU {miou:.3f})")

# ========== MAIN TRAINING FUNCTION ==========
def main_unet():
    p = argparse.ArgumentParser(
        description="Improved U-Net training with advanced features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data_roots", nargs="+", type=str,
                   help="List of VOC roots")
    p.add_argument("--labelmap", type=str, help="Labelmap file")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="models")
    p.add_argument("--architecture", type=str, default="unet", 
                   choices=["unet", "attention_unet", "unet_backbone"],
                   help="Model architecture")
    p.add_argument("--backbone", type=str, default="efficientnet",
                   choices=["efficientnet"],
                   help="Backbone for unet_backbone architecture")
    p.add_argument("--backbone_name", type=str, default="EfficientNetB0",
                   help="Specific backbone model name")
    p.add_argument("--loss", type=str, default="ce",
                   choices=["ce", "focal", "tversky"],
                   help="Loss function")
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--tversky_alpha", type=float, default=0.5)
    p.add_argument("--tversky_beta", type=float, default=0.5)
    p.add_argument("--use_advanced_aug", action="store_true",
                   help="Use advanced data augmentation")
    
    # Xử lý trường hợp chạy trong Colab/Jupyter (có thêm arguments không mong muốn)
    # Colab có thể thêm -f kernel.json vào command line
    known_args, unknown_args = p.parse_known_args()
    
    # Phát hiện môi trường Colab
    import os
    is_colab = os.path.exists("/content") or "COLAB_GPU" in os.environ
    
    # Nếu không có arguments hoặc chỉ có unknown args (Colab kernel), dùng defaults
    if len(sys.argv) == 1 or (len(unknown_args) > 0 and not known_args.data_roots):
        if is_colab:  # Đang chạy trên Colab
            # Tự động tìm labelmap ở các vị trí có thể
            labelmap_candidates = [
                "/content/labelmap.txt",  # Upload trực tiếp
                "/content/drive/MyDrive/SegmentImg/labelmap.txt",  # Trên Drive
            ]
            labelmap_path = "/content/labelmap.txt"  # Default
            for candidate in labelmap_candidates:
                if os.path.exists(candidate):
                    labelmap_path = candidate
                    break
            
            p.set_defaults(
                data_roots=[
                    "/content/drive/MyDrive/SegmentImg/data/cheetah",
                    "/content/drive/MyDrive/SegmentImg/data/lion",
                    "/content/drive/MyDrive/SegmentImg/data/wolf",
                    "/content/drive/MyDrive/SegmentImg/data/tiger",
                    "/content/drive/MyDrive/SegmentImg/data/hyena",
                    "/content/drive/MyDrive/SegmentImg/data/fox",
                ],
                labelmap=labelmap_path,
                epochs=50,
                batch_size=8,  # T4 GPU: 8-16, A100: 16-32
                crop_size=512,
                architecture="attention_unet",
                loss="focal",
                use_advanced_aug=True,
                save_dir="/content/drive/MyDrive/SegmentImg/models",
            )
            print(f"🌐 Running on Google Colab")
            print(f"📁 Labelmap: {labelmap_path}")
            print(f"💾 Models will be saved to: /content/drive/MyDrive/SegmentImg/models")
        else:  # Local Windows
            p.set_defaults(
                data_roots=[
                    r"D:\animal_data\data\cheetah",
                    r"D:\animal_data\data\lion",
                    r"D:\animal_data\data\wolf",
                    r"D:\animal_data\data\tiger",
                    r"D:\animal_data\data\hyena",
                    r"D:\animal_data\data\fox",
                ],
                labelmap=r"D:\animal_data\img_segment\labelmap.txt",
                epochs=5,
                batch_size=4,
                crop_size=512,
                architecture="attention_unet",
                loss="focal",
                use_advanced_aug=True,
                save_dir="models",
            )
            print(f"💻 Running on Local Machine")
        args = p.parse_args(unknown_args)  # Parse với unknown args để ignore chúng
    else:
        args = known_args
        if not args.data_roots or not args.labelmap:
            p.error("--data_roots and --labelmap are required")
    
    roots: List[str] = []
    for item in (args.data_roots or []):
        roots.extend([s for s in item.split(",") if s])
    if not roots:
        p.error("No valid data roots provided")
    
    # Kiểm tra và hiển thị thông tin về paths
    print("\n" + "="*60)
    print("CHECKING DATA PATHS")
    print("="*60)
    missing_roots = []
    for r in roots:
        jp = Path(r) / "JPEGImages"
        sp = Path(r) / "SegmentationClass"
        ip = Path(r) / "ImageSets" / "Segmentation"
        if jp.exists() and sp.exists() and ip.exists():
            n_images = len(list(jp.glob("*"))) if jp.exists() else 0
            print(f"✅ {Path(r).name}: {n_images} images")
        else:
            print(f"❌ {Path(r).name}: Missing VOC folders")
            missing_roots.append(r)
    
    if missing_roots:
        print(f"\n⚠️  Warning: {len(missing_roots)} dataset(s) not found!")
        if is_colab:
            print("💡 Tip: Make sure you have:")
            print("   1. Mounted Google Drive: drive.mount('/content/drive')")
            print("   2. Uploaded data to: /content/drive/MyDrive/SegmentImg/data/")
        print("\nContinue anyway? (y/n): ", end="")
        # Trong Colab, tự động continue
        if not is_colab:
            response = input().strip().lower()
            if response != 'y':
                p.error("Please check your data paths and try again")
    
    # Kiểm tra labelmap
    labelmap_path = Path(args.labelmap)
    if not labelmap_path.exists():
        print(f"\n❌ Labelmap not found: {args.labelmap}")
        if is_colab:
            print("💡 Tip: Upload labelmap.txt to /content/ or /content/drive/MyDrive/SegmentImg/")
        p.error(f"Labelmap file not found: {args.labelmap}")
    else:
        print(f"✅ Labelmap: {args.labelmap}")
    
    # Kiểm tra và tạo save directory
    save_dir_path = Path(args.save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"💾 Save directory: {args.save_dir}")
    print("="*60 + "\n")
    
    set_seed(args.seed)
    
    # Verify mixed precision policy is float32 (should already be set at module level)
    if MIXED_PRECISION_AVAILABLE:
        try:
            current_policy = str(mixed_precision.global_policy())
            if 'float32' in current_policy.lower():
                print(f"✅ Mixed precision policy: {current_policy} (prevents dtype conflicts)")
            else:
                mixed_precision.set_global_policy('float32')
                print(f"⚠️  Warning: Policy was {current_policy}, changed to float32")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify mixed precision policy: {e}")
    else:
        print("✅ TensorFlow default dtype: float32")
    
    names, colors = read_labelmap(Path(args.labelmap))
    num_classes = len(names)
    print(f"Classes ({num_classes}): {names}")
    
    # Build datasets
    train_ds_wrap = EnhancedMultiRootVOCDataset(
        roots=roots, image_set="train",
        names=names, colors=colors,
        crop_size=args.crop_size,
        use_advanced_aug=args.use_advanced_aug
    )
    val_ds_wrap = EnhancedMultiRootVOCDataset(
        roots=roots, image_set="val",
        names=names, colors=colors,
        crop_size=args.crop_size,
        use_advanced_aug=False
    )
    
    train_ds = make_tf_dataset(train_ds_wrap, batch_size=args.batch_size, shuffle=True, ignore_index=255)
    val_ds = make_tf_dataset(val_ds_wrap, batch_size=1, shuffle=False, ignore_index=255)
    
    # Build model
    if args.architecture == "unet":
        model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)
    elif args.architecture == "attention_unet":
        model = build_attention_unet(num_classes=num_classes, dropout=0.2)
    elif args.architecture == "unet_backbone":
        model = build_unet_with_backbone(num_classes=num_classes, backbone=args.backbone, 
                                        backbone_name=args.backbone_name, dropout=0.2)
    else:
        model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)
    
    # CRITICAL: Ensure all model weights are float32 to prevent mixed precision issues
    # This is especially important for EfficientNet backbones loaded from pretrained weights
    if args.architecture == "unet_backbone":
        print("🔄 Final verification: Ensuring all model weights are float32...")
        # Build model first to ensure weights are created
        dummy_input = tf.zeros((1, args.crop_size, args.crop_size, 3), dtype=tf.float32)
        _ = model(dummy_input, training=False)  # Forward pass to build weights
        
        # Convert all trainable weights to float32 (including decoder layers)
        weights_converted = 0
        for layer in model.layers:
            if hasattr(layer, 'weights') and layer.weights:
                for weight in layer.weights:
                    if weight.dtype != tf.float32:
                        weight_value = tf.cast(weight.value(), tf.float32)
                        weight.assign(weight_value)
                        weights_converted += 1
        
        if weights_converted > 0:
            print(f"✅ Converted {weights_converted} weights to float32")
        else:
            print("✅ All weights already float32")
        
        # Verify policy one more time before compilation
        if MIXED_PRECISION_AVAILABLE:
            try:
                current_policy = str(mixed_precision.global_policy())
                if 'float32' not in current_policy.lower():
                    mixed_precision.set_global_policy('float32')
                    print(f"⚠️  Policy was {current_policy}, reset to float32")
                else:
                    print(f"✅ Mixed precision policy confirmed: {current_policy}")
            except:
                pass
    
    # Setup losses
    if args.loss == "ce":
        sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)
    elif args.loss == "focal":
        sem_loss = focal_loss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=255, from_logits=True)
    elif args.loss == "tversky":
        sem_loss = tversky_loss(alpha=args.tversky_alpha, beta=args.tversky_beta, ignore_index=255, from_logits=True)
    else:
        sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)
    
    bce_logits = keras.losses.BinaryCrossentropy(from_logits=True)
    
    losses = {
        "sem_logits": sem_loss,
        "boundary_logits": bce_logits
    }
    loss_weights = {"sem_logits": 1.0, "boundary_logits": 1.0}
    
    optimizer = keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
    
    # Callbacks
    ckpt_path = Path(args.save_dir) / f"{args.architecture}_{args.loss}_best.keras"
    eval_cb = EvalCallback(val_ds, num_classes=num_classes, ignore_index=255, ckpt_path=ckpt_path)
    
    # Train
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        callbacks=[eval_cb],
        verbose=1
    )
    
    print(f"\nTraining completed! Best model saved to: {ckpt_path}")

if __name__ == "__main__":
    main_unet()

