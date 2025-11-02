# Improved Model Training Script with Advanced Features
# Based on model_train_v2.py with enhancements:
# - Advanced data augmentation
# - Class imbalance handling (weights/Focal Loss)
# - Attention U-Net, U-Net++ architectures
# - Stronger backbones (EfficientNet)
# - Multiple loss functions (Weighted CE, Focal, Tversky)
# - Deep supervision
# - TTA (Test Time Augmentation) for inference
# - Post-processing pipeline (morphology, connected components, CRF)

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
from skimage.segmentation import find_boundaries
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import binary_opening, binary_closing, disk
from skimage.measure import label, regionprops
try:
    import pydensecrf.densecrf as densecrf
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    # Warning chỉ hiện 1 lần để không spam console
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    # print("Warning: pydensecrf not available. CRF post-processing will be disabled.")
    # print("To install: !pip install git+https://github.com/lucasb-eyer/pydensecrf.git")

# Try to import EfficientNet/ResNeXt backbones
try:
    from keras.applications import EfficientNetB0, EfficientNetB3, EfficientNetB4
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("Warning: EfficientNet not available. Using standard encoder.")

try:
    from keras.applications import ResNet50, ResNet101
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False
    print("Warning: ResNet not available. Using standard encoder.")

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

# ========== CLASS IMBALANCE HANDLING ==========
def compute_class_weights(masks: List[np.ndarray], num_classes: int, ignore_index: int = 255) -> np.ndarray:
    """Compute class weights from training masks"""
    total_pixels = 0
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    for mask in masks:
        valid_mask = mask != ignore_index
        total_pixels += valid_mask.sum()
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum()
    
    # Inverse frequency weighting
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    
    return class_weights.astype(np.float32)

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

def weighted_sparse_ce_ignore_index(class_weights: np.ndarray, ignore_index: int, from_logits: bool = True):
    """Weighted SparseCategoricalCrossentropy"""
    weights_tf = tf.constant(class_weights, dtype=tf.float32)
    sce = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction='none')
    
    def loss(y_true, y_pred):
        mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        per_px = sce(y_true_clean, y_pred)
        
        # Apply class weights
        y_true_flat = tf.reshape(y_true_clean, [-1])
        weights_flat = tf.gather(weights_tf, y_true_flat)
        weights_flat = tf.reshape(weights_flat, tf.shape(per_px))
        per_px = per_px * weights_flat
        
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
    x = layers.Conv2D(n_filters, 3, padding="same", kernel_initializer="he_normal", use_bias=not use_bn)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(n_filters, 3, padding="same", kernel_initializer="he_normal", use_bias=not use_bn)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
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

def build_unet_plusplus(input_shape=(512, 512, 3), num_classes=6, dropout=0.2, use_batchnorm=True, deep_supervision=True):
    """U-Net++ architecture with deep supervision"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x00 = inputs
    f1, p1 = downsample_block(x00, 64, dropout, use_batchnorm)
    x10 = f1
    
    f2, p2 = downsample_block(p1, 128, dropout, use_batchnorm)
    x20 = f2
    
    f3, p3 = downsample_block(p2, 256, dropout, use_batchnorm)
    x30 = f3
    
    f4, p4 = downsample_block(p3, 512, dropout, use_batchnorm)
    x40 = f4
    
    # Bottleneck
    x50 = double_conv_block(p4, 1024, use_bn=use_batchnorm)
    
    # Dense connections (U-Net++)
    x01 = x00  # skip
    # x11: upsample x10 and concatenate with x01
    x11_up = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x10)
    x11_up = layers.Concatenate(axis=-1)([x11_up, x01])
    x11_up = layers.Dropout(dropout)(x11_up)
    x11 = double_conv_block(x11_up, 64, use_bn=use_batchnorm)
    
    # x21: upsample x20 and concatenate with x11
    x21_up = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x20)
    x21_up = layers.Concatenate(axis=-1)([x21_up, x11])
    x21_up = layers.Dropout(dropout)(x21_up)
    x21 = double_conv_block(x21_up, 128, use_bn=use_batchnorm)
    
    # x31: upsample x30 and concatenate with x21
    x31_up = layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding="same")(x30)
    x31_up = layers.Concatenate(axis=-1)([x31_up, x21])
    x31_up = layers.Dropout(dropout)(x31_up)
    x31 = double_conv_block(x31_up, 256, use_bn=use_batchnorm)
    
    # x41: upsample x40 and concatenate with x31
    x41_up = layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding="same")(x40)
    x41_up = layers.Concatenate(axis=-1)([x41_up, x31])
    x41_up = layers.Dropout(dropout)(x41_up)
    x41 = double_conv_block(x41_up, 512, use_bn=use_batchnorm)
    
    # x51: upsample x50 and concatenate with x41
    x51_up = layers.Conv2DTranspose(1024, kernel_size=3, strides=2, padding="same")(x50)
    x51_up = layers.Concatenate(axis=-1)([x51_up, x41])
    x51_up = layers.Dropout(dropout)(x51_up)
    x51 = double_conv_block(x51_up, 1024, use_bn=use_batchnorm)
    
    # Deep supervision outputs
    outputs = []
    if deep_supervision:
        d1 = layers.Conv2D(num_classes, 1, padding="same", name="ds1")(x21)
        d2 = layers.Conv2D(num_classes, 1, padding="same", name="ds2")(x31)
        d3 = layers.Conv2D(num_classes, 1, padding="same", name="ds3")(x41)
        outputs.extend([d1, d2, d3])
    
    sem_logits = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(x51)
    boundary_logits = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(x51)
    
    outputs.extend([sem_logits, boundary_logits])
    model = Model(inputs, outputs, name="UNetPlusPlus")
    return model

def build_unet_with_backbone(input_shape=(512, 512, 3), num_classes=6, backbone="efficientnet", backbone_name="EfficientNetB0", dropout=0.2):
    """U-Net with pretrained backbone encoder"""
    inputs = layers.Input(shape=input_shape)
    
    # Get backbone
    if backbone == "efficientnet" and EFFICIENTNET_AVAILABLE:
        if backbone_name == "EfficientNetB0":
            backbone_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
        elif backbone_name == "EfficientNetB3":
            backbone_model = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
        elif backbone_name == "EfficientNetB4":
            backbone_model = EfficientNetB4(include_top=False, weights="imagenet", input_tensor=inputs)
        else:
            backbone_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
        
        # Get skip connections from EfficientNet
        # EfficientNet structure: block1, block2, block3, block4, block5, block6, block7
        skip_layers = []
        for block_name in ["block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation"]:
            try:
                skip_layers.append(backbone_model.get_layer(block_name).output)
            except:
                pass
        encoder_output = backbone_model.output
    elif backbone == "resnet" and RESNET_AVAILABLE:
        if backbone_name == "ResNet50":
            backbone_model = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
        elif backbone_name == "ResNet101":
            backbone_model = ResNet101(include_top=False, weights="imagenet", input_tensor=inputs)
        else:
            backbone_model = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
        
        # Get skip connections for ResNet
        skip_layers = [
            backbone_model.get_layer("conv2_block3_out").output,
            backbone_model.get_layer("conv3_block4_out").output,
            backbone_model.get_layer("conv4_block6_out").output,
        ]
        encoder_output = backbone_model.output
    else:
        # Fallback to standard encoder
        return build_unet_with_boundary(input_shape, num_classes, dropout)
    
    # Decoder
    x = encoder_output
    for i, skip in enumerate(reversed(skip_layers)):
        n_filters = 512 // (2 ** i)
        x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding="same")(x)
        x = layers.Concatenate(axis=-1)([x, skip])
        x = double_conv_block(x, n_filters, use_bn=True)
        x = layers.Dropout(dropout)(x)
    
    # Final upsampling
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
    x = double_conv_block(x, 64, use_bn=True)
    
    sem_logits = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(x)
    boundary_logits = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(x)
    
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
        
        distance = ndi.distance_transform_edt(mask)
        coords = peak_local_max(distance, footprint=np.ones((3,3)), labels=mask)
        markers = np.zeros_like(distance, dtype=np.int32)
        for i, (r,c) in enumerate(coords, start=1):
            markers[r, c] = i
        
        if markers.max() == 0:
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
        
        print(f"[Eval] valPA={pixacc:.3f}  valmIoU={miou:.3f}  valBCE={val_bce:.4f}  per-class IoU={np.round(class_ious,3)}")
        
        if miou > self.best_miou:
            self.best_miou = miou
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.ckpt_path.as_posix())
            print(f"Saved best to {self.ckpt_path} (mIoU {miou:.3f})")

# ========== POST-PROCESSING ==========
class PostProcessor:
    """Post-processing pipeline for segmentation masks"""
    def __init__(self, 
                 use_morphology=True,
                 morphology_size=3,
                 min_blob_size=100,
                 use_crf=False,
                 crf_iters=5):
        self.use_morphology = use_morphology
        self.morphology_size = morphology_size
        self.min_blob_size = min_blob_size
        self.use_crf = use_crf and CRF_AVAILABLE
        self.crf_iters = crf_iters
    
    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to smooth mask"""
        if not self.use_morphology:
            return mask
        
        selem = disk(self.morphology_size)
        # Opening to remove small noise
        mask = binary_opening(mask > 0, selem).astype(mask.dtype)
        # Closing to fill small holes
        mask = binary_closing(mask > 0, selem).astype(mask.dtype)
        return mask
    
    def filter_small_blobs(self, mask: np.ndarray) -> np.ndarray:
        """Remove small connected components"""
        if self.min_blob_size <= 0:
            return mask
        
        labeled = label(mask > 0)
        props = regionprops(labeled)
        
        filtered_mask = np.zeros_like(mask)
        for prop in props:
            if prop.area >= self.min_blob_size:
                filtered_mask[labeled == prop.label] = mask[labeled == prop.label]
        
        return filtered_mask
    
    def apply_crf(self, image: np.ndarray, mask_probs: np.ndarray) -> np.ndarray:
        """Apply DenseCRF for boundary refinement"""
        if not self.use_crf:
            return np.argmax(mask_probs, axis=-1)
        
        H, W = image.shape[:2]
        num_classes = mask_probs.shape[-1]
        
        # Prepare unary potentials
        unary = -np.log(mask_probs + 1e-8)
        unary = unary.reshape((num_classes, -1))
        
        # Create CRF
        d = densecrf.DenseCRF2D(W, H, num_classes)
        d.setUnaryEnergy(unary)
        
        # Add pairwise potentials
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
        
        # Inference
        Q = d.inference(self.crf_iters)
        map_result = np.argmax(Q, axis=0).reshape((H, W))
        
        return map_result
    
    def process(self, image: np.ndarray, mask: np.ndarray, mask_probs: np.ndarray = None) -> np.ndarray:
        """Full post-processing pipeline"""
        result = mask.copy()
        
        # Morphology
        result = self.apply_morphology(result)
        
        # Filter small blobs
        result = self.filter_small_blobs(result)
        
        # CRF refinement (if enabled and probabilities available)
        if self.use_crf and mask_probs is not None:
            result = self.apply_crf(image, mask_probs)
        
        return result

# ========== TEST TIME AUGMENTATION ==========
class TTAInference:
    """Test Time Augmentation for inference"""
    def __init__(self, model, tta_transforms=["flip_h", "flip_v", "rotate_90"]):
        self.model = model
        self.tta_transforms = tta_transforms
    
    def _apply_transform(self, img: np.ndarray, transform: str) -> np.ndarray:
        """Apply transformation to image"""
        if transform == "flip_h":
            return np.flip(img, axis=1)
        elif transform == "flip_v":
            return np.flip(img, axis=0)
        elif transform == "rotate_90":
            return np.rot90(img, k=1, axes=(0, 1))
        elif transform == "rotate_180":
            return np.rot90(img, k=2, axes=(0, 1))
        elif transform == "rotate_270":
            return np.rot90(img, k=3, axes=(0, 1))
        return img
    
    def _reverse_transform(self, mask: np.ndarray, transform: str) -> np.ndarray:
        """Reverse transformation on mask"""
        if transform == "flip_h":
            return np.flip(mask, axis=1)
        elif transform == "flip_v":
            return np.flip(mask, axis=0)
        elif transform == "rotate_90":
            return np.rot90(mask, k=-1, axes=(0, 1))
        elif transform == "rotate_180":
            return np.rot90(mask, k=-2, axes=(0, 1))
        elif transform == "rotate_270":
            return np.rot90(mask, k=-3, axes=(0, 1))
        return mask
    
    def predict(self, img: np.ndarray) -> np.ndarray:
        """Predict with TTA"""
        # Original prediction
        img_tf = tf.expand_dims(tf.constant(img, dtype=tf.float32), 0)
        outputs = self.model(img_tf, training=False)
        if isinstance(outputs, list):
            sem_logits = outputs[0]
        else:
            sem_logits = outputs.get("sem_logits") if isinstance(outputs, dict) else outputs
        
        probs = tf.nn.softmax(sem_logits, axis=-1).numpy()[0]
        predictions = [probs]
        
        # TTA predictions
        for transform in self.tta_transforms:
            img_transformed = self._apply_transform(img, transform)
            img_tf = tf.expand_dims(tf.constant(img_transformed, dtype=tf.float32), 0)
            outputs = self.model(img_tf, training=False)
            if isinstance(outputs, list):
                sem_logits = outputs[0]
            else:
                sem_logits = outputs.get("sem_logits") if isinstance(outputs, dict) else outputs
            
            probs_transformed = tf.nn.softmax(sem_logits, axis=-1).numpy()[0]
            probs_reversed = self._reverse_transform(probs_transformed, transform)
            predictions.append(probs_reversed)
        
        # Average predictions
        avg_probs = np.mean(predictions, axis=0)
        return np.argmax(avg_probs, axis=-1), avg_probs

# ========== INFERENCE PIPELINE ==========
def inference_pipeline(model_path: str, image_path: str, output_path: str,
                      labelmap_path: str, crop_size: int = 512,
                      use_tta: bool = True, use_postprocessing: bool = True):
    """
    Complete inference pipeline:
    1. Read image → normalize → resize/pad
    2. Model inference (with TTA)
    3. Argmax → resize back to original size
    4. Post-processing (morphology, connected components, CRF)
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load labelmap
    names, colors = read_labelmap(Path(labelmap_path))
    num_classes = len(names)
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # Resize/pad to crop_size
    h, w = img_np.shape[:2]
    scale = min(crop_size / h, crop_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to crop_size
    pad_h = crop_size - new_h
    pad_w = crop_size - new_w
    img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    
    # Inference
    if use_tta:
        tta = TTAInference(model)
        mask_pred, mask_probs = tta.predict(img_padded)
    else:
        img_tf = tf.expand_dims(tf.constant(img_padded, dtype=tf.float32), 0)
        outputs = model(img_tf, training=False)
        if isinstance(outputs, list):
            sem_logits = outputs[0]
        else:
            sem_logits = outputs.get("sem_logits") if isinstance(outputs, dict) else outputs
        mask_probs = tf.nn.softmax(sem_logits, axis=-1).numpy()[0]
        mask_pred = np.argmax(mask_probs, axis=-1)
    
    # Remove padding
    mask_pred = mask_pred[:new_h, :new_w]
    mask_probs = mask_probs[:new_h, :new_w]
    
    # Resize back to original size
    mask_pred = cv2.resize(mask_pred.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
    mask_probs_resized = np.zeros((original_size[1], original_size[0], num_classes), dtype=np.float32)
    for c in range(num_classes):
        mask_probs_resized[:, :, c] = cv2.resize(mask_probs[:, :, c], original_size, interpolation=cv2.INTER_LINEAR)
    
    # Post-processing
    if use_postprocessing:
        post_processor = PostProcessor(use_morphology=True, min_blob_size=100, use_crf=False)
        img_original = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask_pred = post_processor.process(img_original, mask_pred, mask_probs_resized)
    
    # Convert to color mask
    color_mask = np.zeros((mask_pred.shape[0], mask_pred.shape[1], 3), dtype=np.uint8)
    for i, (name, color) in enumerate(zip(names, colors)):
        color_mask[mask_pred == i] = color
    
    # Save
    Image.fromarray(color_mask).save(output_path)
    print(f"Saved prediction to {output_path}")

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
                   choices=["unet", "attention_unet", "unet_plusplus", "unet_backbone"],
                   help="Model architecture")
    p.add_argument("--backbone", type=str, default="efficientnet",
                   choices=["efficientnet", "resnet"],
                   help="Backbone for unet_backbone architecture")
    p.add_argument("--backbone_name", type=str, default="EfficientNetB0",
                   help="Specific backbone model name")
    p.add_argument("--loss", type=str, default="ce",
                   choices=["ce", "weighted_ce", "focal", "tversky"],
                   help="Loss function")
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--tversky_alpha", type=float, default=0.5)
    p.add_argument("--tversky_beta", type=float, default=0.5)
    p.add_argument("--deep_supervision", action="store_true",
                   help="Enable deep supervision for U-Net++")
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
    
    # Compute class weights if using weighted loss
    class_weights = None
    if args.loss == "weighted_ce":
        print("Computing class weights...")
        masks = []
        for i in range(min(100, len(train_ds_wrap))):  # Sample for efficiency
            _, mask = train_ds_wrap.get_item(i)
            masks.append(mask)
        class_weights = compute_class_weights(masks, num_classes, ignore_index=255)
        print(f"Class weights: {class_weights}")
    
    train_ds = make_tf_dataset(train_ds_wrap, batch_size=args.batch_size, shuffle=True, ignore_index=255)
    val_ds = make_tf_dataset(val_ds_wrap, batch_size=1, shuffle=False, ignore_index=255)
    
    # Build model
    if args.architecture == "unet":
        model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)
    elif args.architecture == "attention_unet":
        model = build_attention_unet(num_classes=num_classes, dropout=0.2)
    elif args.architecture == "unet_plusplus":
        model = build_unet_plusplus(num_classes=num_classes, dropout=0.2, deep_supervision=args.deep_supervision)
    elif args.architecture == "unet_backbone":
        model = build_unet_with_backbone(num_classes=num_classes, backbone=args.backbone, 
                                        backbone_name=args.backbone_name, dropout=0.2)
    else:
        model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)
    
    # Setup losses
    if args.loss == "ce":
        sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)
    elif args.loss == "weighted_ce":
        sem_loss = weighted_sparse_ce_ignore_index(class_weights, ignore_index=255, from_logits=True)
    elif args.loss == "focal":
        sem_loss = focal_loss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=255, from_logits=True)
    elif args.loss == "tversky":
        sem_loss = tversky_loss(alpha=args.tversky_alpha, beta=args.tversky_beta, ignore_index=255, from_logits=True)
    else:
        sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)
    
    bce_logits = keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Handle multiple outputs for deep supervision
    if args.architecture == "unet_plusplus" and args.deep_supervision:
        losses = {
            "ds1": sem_loss,
            "ds2": sem_loss,
            "ds3": sem_loss,
            "sem_logits": sem_loss,
            "boundary_logits": bce_logits
        }
        loss_weights = {
            "ds1": 0.25,
            "ds2": 0.25,
            "ds3": 0.25,
            "sem_logits": 1.0,
            "boundary_logits": 1.0
        }
    else:
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

