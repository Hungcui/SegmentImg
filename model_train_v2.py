import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import random

import keras
import numpy as np
from PIL import Image

# === Removed all torch/torchvision imports ===
import tensorflow as tf
from keras import layers, Model, ops
from skimage.segmentation import find_boundaries

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

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
    Ignores blank lines and lines starting with '#'.
    Returns (names, colors).
    """
    names, colors = [], []
    text = Path(labelmap_path).read_text(encoding="utf-8").splitlines()
    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Check again labelmap line (no colon): {line}")

        # Split once to get name, then keep the rest for color, action, ...
        name, rest = line.split(":", 1)
        name = name.strip()

        # Take 1st field with index 0 (color only)
        color_field = rest.split(":", 1)[0]
        comps = color_field.split(",")

        if len(comps) != 3:
            raise ValueError(f"RGB must have 3 components: {line}")

        try:
            r, g, b = [int(c.strip()) for c in comps]
        except Exception as e:
            raise ValueError(f"Non-integer RGB in line: {line}") from e

        names.append(name)
        colors.append((r, g, b))

    return names, colors

def build_color_to_index(colors: List[Tuple[int,int,int]]) -> Dict[Tuple[int,int,int], int]:
    '''
    - input: [(0,0,0), (224,64,64), (160,96,64)]
    - output: {(0, 0, 0): 0, (224, 64, 64): 1, (160, 96, 64): 2}
    - FOR WHAT ?
    '''
    return {tuple(map(int, c)): i for i, c in enumerate(colors)}

def mask_rgb_to_index(mask_img: Image.Image, color_to_index: Dict[Tuple[int,int,int], int], ignore_index=255) -> np.ndarray:
    """
    Convert an RGB palette/truecolor mask (H,W,3) into class indices (H,W).
    Any pixel color not found in color_to_index becomes ignore_index.
    """
    m = np.array(mask_img.convert("RGB"), dtype=np.uint8)  # (H,W,3)
    h, w, _ = m.shape
    flat = m.reshape(-1, 3)
    out = np.full((h*w,), ignore_index, dtype=np.uint8)
    # Build a lookup by packing RGB into 24-bit int for speed
    keys = (flat[:,0].astype(np.int32) << 16) | (flat[:,1].astype(np.int32) << 8) | flat[:,2].astype(np.int32)
    lut = {}
    for (r,g,b), idx in color_to_index.items():
        lut[(r<<16) | (g<<8) | b] = idx
    # map
    for k, idx in lut.items():
        out[keys == k] = idx
    return out.reshape(h, w)

def set_seed(seed: int = 0):
    '''
    fix value of seed for random
    '''
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# NumPy versions of metric
def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int=255):
    # pred, target: (N,H,W) int64/int32
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    k = (target.astype(np.int64) * num_classes + pred.astype(np.int64))
    bincount = np.bincount(k, minlength=num_classes**2)
    return bincount.reshape(num_classes, num_classes)

def miou_from_confmat(cm: np.ndarray) -> Tuple[float, List[float]]:
    # cm[c,t] = pred c, true t
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=1).astype(np.float64) - tp
    fn = cm.sum(axis=0).astype(np.float64) - tp
    denom = tp + fp + fn + 1e-6
    iou = (tp / denom)
    mean_iou = float(np.mean(iou))
    return mean_iou, list(map(float, iou))

# U-Net model
def double_conv_block(x, n_filters, use_bn=True):
    # Conv -> BN -> ReLU -> Conv -> BN -> ReLU (same padding keeps H,W)
    x = layers.Conv2D(n_filters, 3, padding="same",
                      kernel_initializer="he_normal", use_bias=not use_bn)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(n_filters, 3, padding="same",
                      kernel_initializer="he_normal", use_bias=not use_bn)(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def downsample_block(x, n_filters, dropout=0.2, use_bn=True):
    """Return (f, p): features before pool (skip), and pooled+dropout tensor."""
    f = double_conv_block(x, n_filters, use_bn=use_bn)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout)(p)
    return f, p

def upsample_block(x, conv_feature, n_filters, dropout=0.2, use_bn=True):
    """Transpose conv upsample (x2), concatenate skip, dropout, then double conv."""
    x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding="same")(x)
    x = layers.Concatenate(axis=-1)([x, conv_feature])
    x = layers.Dropout(dropout)(x)
    x = double_conv_block(x, n_filters, use_bn=use_bn)
    return x

# U-Net with boundary head
def build_unet_with_boundary(input_shape=(512, 512, 3), num_classes=6, 
                             dropout=0.2, use_batchnorm=True):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    f1, p1 = downsample_block(inputs, 64, dropout, use_batchnorm)   
    f2, p2 = downsample_block(p1, 128, dropout, use_batchnorm)   
    f3, p3 = downsample_block(p2, 256, dropout, use_batchnorm)  
    f4, p4 = downsample_block(p3, 512, dropout, use_batchnorm) 

    # Bottleneck
    bottleneck = double_conv_block(p4, 1024,  use_bn=use_batchnorm) 

    # Decoder
    u6 = upsample_block(bottleneck, f4, 512, dropout, use_batchnorm)  
    u7 = upsample_block(u6, f3, 256, dropout, use_batchnorm) 
    u8 = upsample_block(u7, f2, 128, dropout, use_batchnorm)
    u9 = upsample_block(u8, f1, 64, dropout, use_batchnorm)

    # Heads (logits)
    sem_logits = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(u9)     # (H,W,C)
    boundary_logits = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(u9)     # (H,W,1)

    model = Model(inputs, [sem_logits, boundary_logits], name="UNetBoundary")
    return model

def make_boundary_targets(mask_batch: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """
    mask_batch: (N, H, W) int class indices.
    Returns boundary targets in shape (N, 1, H, W) float32 in {0,1}, with ignore as 0.
    Uses skimage.segmentation.find_boundaries on the *semantic* mask.
    """
    out = []
    for m in mask_batch:
        mm = m.copy()
        mm[mm == ignore_index] = 0
        b = find_boundaries(mm, mode="inner").astype(np.float32)  # (H,W) True at inner boundaries
        out.append(b[..., None])  # (H,W,1) channel-last
    return np.stack(out, axis=0)  # (N,H,W,1)

def instances_from_sem_and_boundary(
    sem_logits,        # (1,C,H,W) or (H,W,C)
    boundary_logits,   # (1,1,H,W) or (H,W,1)
    thing_class_ids: List[int],      # e.g., [1,2,3,...] (exclude background=0)
    sem_thresh: float = 0.5,
    boundary_thresh: float = 0.5,
) -> np.ndarray:
    """
    Returns a (H,W) np.int32 labeled instance mask where each instance has a unique >0 id.
    Labels are unique across classes (we offset while stacking).
    """
    # Accept tf or np; standardize to np with (C,H,W)/(1,H,W) layout
    if isinstance(sem_logits, tf.Tensor): sem_logits = sem_logits.numpy()
    if isinstance(boundary_logits, tf.Tensor): boundary_logits = boundary_logits.numpy()

    # Standardize layout to (C,H,W) and (H,W)
    if sem_logits.ndim == 4:   # (1,C,H,W) or (1,H,W,C)
        if sem_logits.shape[0] == 1 and sem_logits.shape[1] != 1:
            sem_prob = tf.nn.softmax(sem_logits, axis=1).numpy()[0]            # (C,H,W)
        else:
            sem_prob = tf.nn.softmax(np.transpose(sem_logits, (0,3,1,2)), axis=1).numpy()[0]
    else:  # (H,W,C)
        sem_prob = tf.nn.softmax(np.transpose(sem_logits, (2,0,1)), axis=0).numpy()

    if boundary_logits.ndim == 4:
        if boundary_logits.shape[-1] == 1:  # (1,H,W,1)
            boundary_prob = tf.nn.sigmoid(boundary_logits)[0,...,0].numpy()
        else:                               # (1,1,H,W)
            boundary_prob = tf.nn.sigmoid(boundary_logits)[0,0].numpy()
    else:  # (H,W,1)
        boundary_prob = tf.nn.sigmoid(boundary_logits[...,0]).numpy()

    H, W = boundary_prob.shape
    instance_map = np.zeros((H, W), dtype=np.int32)
    cur_label = 0

    for cid in thing_class_ids:
        fg = sem_prob[cid]  # (H,W)
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

def make_unet_model(num_classes: int):
    return build_unet_with_boundary(num_classes=num_classes)

# Data wrapper
class MultiRootVOCDataset:
    """
    Read VOC-style segmentation from multiple dataset roots.
    Each root must contain:
      JPEGImages/, SegmentationClass/, ImageSets/Segmentation/train.txt or val.txt
    A single, unified (names, colors) defines the global classes.
    """
    def __init__(self, roots: List[str], image_set: str,
                 names: List[str], colors: List[Tuple[int,int,int]],
                 crop_size: int = 512, random_scale=(0.5, 2.0),
                 hflip_prob: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.roots = [Path(r) for r in roots]
        self.image_set = image_set
        self.names, self.colors = names, colors
        self.ignore_index = ignore_index
        self.crop_size, self.random_scale, self.hflip_prob = crop_size, random_scale, hflip_prob
        self.color_to_index = build_color_to_index(colors)

        # Build list of (root, id), look like 
        # [
        #   (Path(".../cheetah"), "2008_000123"),
        #   (Path(".../cheetah"), "2008_000456"),
        #   (Path(".../lion"),    "2011_003210"),
        #   ...
        # ]
        self.samples = []
        for root in self.roots:
            set_file = root / "ImageSets" / "Segmentation" / f"{image_set}.txt"
            ids = [s.strip() for s in set_file.read_text().splitlines() if s.strip()]
            for img_id in ids:
                self.samples.append((root, img_id))

        # Normalization (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
        mask = mask_rgb_to_index(mask_rgb, self.color_to_index, ignore_index=self.ignore_index)  # (H,W) uint8
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
        # Pad if needed
        if img.height < th or img.width < tw:
            pad_h, pad_w = max(0, th - img.height), max(0, tw - img.width)
            # left, top, right, bottom
            img = Image.fromarray(np.pad(np.array(img),
                                         ((0,pad_h),(0,pad_w),(0,0)),
                                         mode="constant", constant_values=0).astype(np.uint8))
            mask = np.pad(mask, ((0,pad_h),(0,pad_w)),
                          mode="constant", constant_values=self.ignore_index)

        # Random crop
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
        # center crop
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
        else:
            img, mask = self._center_crop_or_resize(img, mask)

        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        # to (H,W,3)
        mask_np = mask.astype(np.int64)
        return img_np, mask_np

# Losses 
def sparse_ce_ignore_index(ignore_index: int, from_logits: bool = True):
    """
    SparseCategoricalCrossentropy that masks out ignore_index.
    y_true: (B,H,W) int
    y_pred: (B,H,W,C) logits
    """
    sce = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, reduction='none')  
    def loss(y_true, y_pred):
        # y_true: (B,H,W), y_pred: (B,H,W,C)
        mask = tf.not_equal(y_true, ignore_index)                       # (B,H,W)
        # Keras expects labels >=0; replace ignore with 0 for loss computation but will be masked
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        per_px = sce(y_true_clean, y_pred)                              # (B,H,W)
        per_px = tf.where(mask, per_px, tf.zeros_like(per_px))
        denom = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        return tf.reduce_sum(per_px) / denom
    return loss

# Binary cross-entropy with logits for boundary head
bce_logits = keras.losses.BinaryCrossentropy(from_logits=True)

# tf.data pipelines
def make_tf_dataset(voc: MultiRootVOCDataset, batch_size: int, shuffle: bool, ignore_index: int):
    indices = np.arange(len(voc), dtype=np.int32)

    def _py_load(idx):
        img, mask = voc.get_item(int(idx))
        # returns (1, H, W, 1), take [0] -> (H, W, 1)
        bt = make_boundary_targets(np.expand_dims(mask, 0), ignore_index=ignore_index)[0]
        return img.astype(np.float32), mask.astype(np.int32), bt.astype(np.float32)

    def _tf_map(idx):
        img, mask, bt = tf.numpy_function(_py_load, [idx], [tf.float32, tf.int32, tf.float32])
        # Set shapes (dynamic H,W but channels known)
        img.set_shape([None, None, 3])
        mask.set_shape([None, None])
        bt.set_shape([None, None, 1])
        return img, {"sem_logits": mask, "boundary_logits": bt}

    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(voc), reshuffle_each_iteration=True)
    ds = ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=shuffle)  # like drop_last for train
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# Evaluation callback (val pixel acc, mIoU, boundary BCE)
class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, num_classes: int, ignore_index: int, ckpt_path: Path):
        super().__init__()
        self.val_ds = val_ds
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.best_miou = 0.0
        self.ckpt_path = ckpt_path

    def on_epoch_end(self, epoch, logs=None):
        # Accumulate confusion matrix
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        correct_px, total_px = 0, 0
        total_bce = 0.0
        n_samples = 0

        for batch in self.val_ds:
            imgs, y = batch
            masks = y["sem_logits"].numpy()          # (B,H,W)
            boundary_t = y["boundary_logits"].numpy()# (B,H,W,1)
            sem_logits, boundary_logits = self.model(imgs, training=False)
            preds = tf.argmax(sem_logits, axis=-1).numpy().astype(np.int64)  # (B,H,W)

            # confusion matrix + pix acc
            cm += compute_confusion_matrix(preds, masks, self.num_classes, self.ignore_index)
            valid = (masks != self.ignore_index)
            correct_px += (preds == masks)[valid].sum()
            total_px += valid.sum()

            # boundary loss (monitor)
            total_bce += bce_logits(boundary_t, boundary_logits).numpy() * imgs.shape[0]
            n_samples += imgs.shape[0]

        pixacc = correct_px / max(1, total_px)
        miou, class_ious = miou_from_confmat(cm)
        val_bce = total_bce / max(1, n_samples)

        print(f"[Eval] valPA={pixacc:.3f}  valmIoU={miou:.3f}  valBCE={val_bce:.4f}  per-class IoU={np.round(class_ious,3)}")

        # Save 
        if miou > self.best_miou:
            self.best_miou = miou
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.ckpt_path.as_posix())  
            print(f"Saved best to {self.ckpt_path} (mIoU {miou:.3f})")

def main_unet():
    p = argparse.ArgumentParser(
        description="U-Net training over multiple VOC-style roots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data_roots", nargs="+", type=str,
                   help="List of VOC roots (each has JPEGImages/, SegmentationClass/, ImageSets/Segmentation/)")
    p.add_argument("--labelmap", type=str,
                   help="Unified labelmap.txt for ALL classes (order defines indices).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="models")

    # If no CLI args, inject default “press Run” 
    if len(sys.argv) == 1:
        p.set_defaults(
            data_roots=[
                r"D:\animal_data\img_segment\data\cheetah",
                r"D:\animal_data\img_segment\data\lion",
                r"D:\animal_data\img_segment\data\wolf",
                r"D:\animal_data\img_segment\data\tiger",
                r"D:\animal_data\img_segment\data\hyena",
                r"D:\animal_data\img_segment\data\fox",
            ],
            labelmap=r"D:\animal_data\img_segment\labelmap.txt",
            epochs=5,
            batch_size=4,
            crop_size=512,
        )
        args = p.parse_args([])
    else:
        args = p.parse_args()
        if not args.data_roots or not args.labelmap:
            p.error("--data_roots and --labelmap are required when passing CLI arguments.")

    # Normalize roots: allow comma-separated entries as well
    roots: List[str] = []
    for item in (args.data_roots or []):
        roots.extend([s for s in item.split(",") if s])
    if not roots:
        p.error("No valid data roots provided.")

    # checks
    for r in roots:
        jp = Path(r) / "JPEGImages"
        sp = Path(r) / "SegmentationClass"
        ip = Path(r) / "ImageSets" / "Segmentation"
        if not (jp.exists() and sp.exists() and ip.exists()):
            p.error(f"Root missing VOC folders: {r}")

    set_seed(args.seed)

    names, colors = read_labelmap(Path(args.labelmap))
    num_classes = len(names)
    print(f"Classes ({num_classes}): {names}")

    # Build datasets
    train_ds_wrap = MultiRootVOCDataset(
        roots=roots, image_set="train",
        names=names, colors=colors,
        crop_size=args.crop_size
    )
    val_ds_wrap = MultiRootVOCDataset(
        roots=roots, image_set="val",
        names=names, colors=colors,
        crop_size=args.crop_size
    )

    train_ds = make_tf_dataset(train_ds_wrap, batch_size=args.batch_size, shuffle=True, ignore_index=255)
    val_ds   = make_tf_dataset(val_ds_wrap, batch_size=1, shuffle=False, ignore_index=255)

    model = make_unet_model(num_classes)

    # Compile with masked sparse CE for semantic head + BCE(logits) for boundary head
    losses = {
        "sem_logits": sparse_ce_ignore_index(ignore_index=255, from_logits=True),  # masked sparse CE
        "boundary_logits": bce_logits
    }
    loss_weights = {"sem_logits": 1.0, "boundary_logits": 1.0}
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    # Eval + save-best callback (mirrors your original loop)
    ckpt_path = Path(r"D:\animal_data\img_segment\models\unet_boundary_best.keras")
    eval_cb = EvalCallback(val_ds, num_classes=num_classes, ignore_index=255, ckpt_path=ckpt_path)

    # Train
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        callbacks=[eval_cb],
        verbose=1
    )

if __name__ == "__main__":
    main_unet()
