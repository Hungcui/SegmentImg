"""
U-Net (semantic + boundary) training over multiple VOC-style roots.
Now with:
- Full-model checkpoint per epoch (.keras) via ModelCheckpoint
- Resume training from latest/specified .keras via --resume
- Serializable custom loss (ignore_index) registered for Keras 3

Refs:
- Keras whole-model .keras save/load (architecture + weights + optimizer state)
- ModelCheckpoint for periodic saves

Run code:
    New training: python model_train.py --epochs 100 --save_dir D:\animal_data\models
    Continue trainint: python train_unet_resume.py --epochs 100 --save_dir D:\animal_data\models --resume auto

"""

# ---- Must be set BEFORE importing keras (Keras 3) ----
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # choose TF backend for Keras 3

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys
import re
import random
import numpy as np
from PIL import Image

import tensorflow as tf
import keras
from keras import layers, Model

from skimage.segmentation import find_boundaries
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ---------- Pillow resampling shim (handles Pillow 9/10+) ----------
from PIL import Image as _PILImage
if hasattr(_PILImage, "Resampling"):
    RESAMPLE_BILINEAR = _PILImage.Resampling.BILINEAR
    RESAMPLE_NEAREST  = _PILImage.Resampling.NEAREST
else:
    RESAMPLE_BILINEAR = _PILImage.BILINEAR
    RESAMPLE_NEAREST  = _PILImage.NEAREST

# ---------- Labelmap ----------
def read_labelmap(labelmap_path: Path):
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
        r, g, b = [int(c.strip()) for c in comps]
        names.append(name)
        colors.append((r, g, b))
    return names, colors

def build_color_to_index(colors: List[Tuple[int,int,int]]) -> Dict[Tuple[int,int,int], int]:
    return {tuple(map(int, c)): i for i, c in enumerate(colors)}

def mask_rgb_to_index(mask_img: Image.Image, color_to_index: Dict[Tuple[int,int,int], int], ignore_index=255) -> np.ndarray:
    m = np.array(mask_img.convert("RGB"), dtype=np.uint8)  # (H,W,3)
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

# ---------- Reproducibility ----------
def set_seed(seed: int = 0, enable_determinism: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if enable_determinism:
        tf.config.experimental.enable_op_determinism(True)

# ---------- Metrics ----------
def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int=255):
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
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

# ---------- U-Net ----------
def double_conv_block(x, n_filters, use_bn=True):
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
    f = double_conv_block(x, n_filters, use_bn=use_bn)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout)(p)
    return f, p

def upsample_block(x, conv_feature, n_filters, dropout=0.2, use_bn=True):
    x = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding="same")(x)
    x = layers.Concatenate(axis=-1)([x, conv_feature])
    x = layers.Dropout(dropout)(x)
    x = double_conv_block(x, n_filters, use_bn=use_bn)
    return x

def build_unet_with_boundary(input_shape=(512, 512, 3), num_classes=6, 
                             dropout=0.2, use_batchnorm=True):
    inputs = layers.Input(shape=input_shape)
    f1, p1 = downsample_block(inputs, 64, dropout, use_batchnorm)   
    f2, p2 = downsample_block(p1, 128, dropout, use_batchnorm)   
    f3, p3 = downsample_block(p2, 256, dropout, use_batchnorm)  
    f4, p4 = downsample_block(p3, 512, dropout, use_batchnorm) 
    bottleneck = double_conv_block(p4, 1024,  use_bn=use_batchnorm) 
    u6 = upsample_block(bottleneck, f4, 512, dropout, use_batchnorm)  
    u7 = upsample_block(u6, f3, 256, dropout, use_batchnorm) 
    u8 = upsample_block(u7, f2, 128, dropout, use_batchnorm)
    u9 = upsample_block(u8, f1, 64, dropout, use_batchnorm)
    sem_logits = layers.Conv2D(num_classes, 1, padding="same", name="sem_logits")(u9)     # (H,W,C)
    boundary_logits = layers.Conv2D(1, 1, padding="same", name="boundary_logits")(u9)     # (H,W,1)
    model = Model(inputs, [sem_logits, boundary_logits], name="UNetBoundary")
    return model

# ---------- Boundary targets ----------
def make_boundary_targets(mask_batch: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    out = []
    for m in mask_batch:
        mm = m.copy()
        mm[mm == ignore_index] = 0
        b = find_boundaries(mm, mode="inner").astype(np.float32)  # (H,W)
        out.append(b[..., None])
    return np.stack(out, axis=0)

# ---------- Instances (not used in training; here for parity) ----------
def instances_from_sem_and_boundary(
    sem_logits,
    boundary_logits,
    thing_class_ids: List[int],
    sem_thresh: float = 0.5,
    boundary_thresh: float = 0.5,
) -> np.ndarray:
    if isinstance(sem_logits, tf.Tensor): sem_logits = sem_logits.numpy()
    if isinstance(boundary_logits, tf.Tensor): boundary_logits = boundary_logits.numpy()
    if boundary_logits.ndim == 4:
        if boundary_logits.shape[0] == 1 and boundary_logits.shape[-1] == 1:       # (1,H,W,1)
            boundary_prob = tf.nn.sigmoid(boundary_logits)[0, ..., 0].numpy()
        elif boundary_logits.shape[0] == 1 and boundary_logits.shape[1] == 1:       # (1,1,H,W)
            boundary_prob = tf.nn.sigmoid(boundary_logits)[0, 0].numpy()
        else:
            raise ValueError(f"Unexpected boundary_logits shape {boundary_logits.shape}")
    elif boundary_logits.ndim == 3 and boundary_logits.shape[-1] == 1:              # (H,W,1)
        boundary_prob = tf.nn.sigmoid(boundary_logits[..., 0]).numpy()
    elif boundary_logits.ndim == 2:                                                 # (H,W)
        boundary_prob = tf.nn.sigmoid(boundary_logits).numpy()
    else:
        raise ValueError(f"Unsupported boundary_logits shape {boundary_logits.shape}")
    H, W = boundary_prob.shape

    if sem_logits.ndim == 4:
        if sem_logits.shape[0] != 1:
            raise ValueError(f"Expected batch=1 for sem_logits, got {sem_logits.shape[0]}")
        sem_logits = sem_logits[0]
    if sem_logits.ndim != 3:
        raise ValueError(f"Unsupported sem_logits shape after squeeze: {sem_logits.shape}")

    if sem_logits.shape[0:2] == (H, W):                 # (H,W,C)
        sem_prob_hwC = tf.nn.softmax(sem_logits, axis=-1).numpy()
        sem_prob = np.moveaxis(sem_prob_hwC, -1, 0)     # (C,H,W)
    elif sem_logits.shape[1:3] == (H, W):               # (C,H,W)
        sem_prob = tf.nn.softmax(sem_logits, axis=0).numpy()
    else:
        raise ValueError("sem_logits spatial dims do not match boundary")

    instance_map = np.zeros((H, W), dtype=np.int32)
    cur_label = 0
    for cid in thing_class_ids:
        if cid < 0 or cid >= sem_prob.shape[0]:
            continue
        fg = sem_prob[cid] >= sem_thresh
        mask = fg & (boundary_prob < boundary_thresh)
        if mask.sum() == 0:
            continue
        distance = ndi.distance_transform_edt(mask)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=mask)
        markers = np.zeros_like(distance, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
        if markers.max() == 0:
            markers, _ = ndi.label(mask)
        labels_ = watershed(-distance, markers, mask=mask).astype(np.int32)
        labels_[labels_ > 0] += cur_label
        instance_map[labels_ > 0] = labels_[labels_ > 0]
        cur_label = instance_map.max()
    return instance_map

def make_unet_model(num_classes: int):
    return build_unet_with_boundary(num_classes=num_classes)

# ---------- Serializable custom loss: Sparse CE with ignore_index ----------
@keras.saving.register_keras_serializable(package="custom")
class SparseCEIgnoreIndex(keras.losses.Loss):
    def __init__(self, ignore_index: int = 255, from_logits: bool = True, name: str = "sparse_ce_ignore_index"):
        super().__init__(name=name)
        self.ignore_index = int(ignore_index)
        self.from_logits = bool(from_logits)
        self._sce = keras.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits, reduction="none")

    def call(self, y_true, y_pred):
        # y_true: (B,H,W) int ; y_pred: (B,H,W,C) logits
        mask = tf.not_equal(y_true, self.ignore_index)  # (B,H,W)
        y_true_clean = tf.where(mask, y_true, tf.zeros_like(y_true))
        per_px = self._sce(y_true_clean, y_pred)        # (B,H,W)
        per_px = tf.where(mask, per_px, tf.zeros_like(per_px))
        denom = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
        return tf.reduce_sum(per_px) / denom

    def get_config(self):
        return {"ignore_index": self.ignore_index, "from_logits": self.from_logits, "name": self.name}

# Keep a BCE for boundary head
bce_logits_no_red = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

# ---------- Dataset wrapper ----------
class MultiRootVOCDataset:
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

        self.samples = []
        for root in self.roots:
            set_file = root / "ImageSets" / "Segmentation" / f"{image_set}.txt"
            ids = [s.strip() for s in set_file.read_text().splitlines() if s.strip()]
            for img_id in ids:
                self.samples.append((root, img_id))

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
            new_w = max(1, int(round(img.width  * s)))
            new_h = max(1, int(round(img.height * s)))
            img = img.resize((new_w, new_h), RESAMPLE_BILINEAR)
            mask_pil = Image.fromarray(mask.astype(np.uint16), mode="I;16")
            mask_pil = mask_pil.resize((new_w, new_h), RESAMPLE_NEAREST)
            mask = np.array(mask_pil, dtype=np.int64)
        return img, mask

    def _random_crop(self, img, mask):
        th, tw = self.crop_size, self.crop_size
        if img.height < th or img.width < tw:
            pad_h, pad_w = max(0, th - img.height), max(0, tw - img.width)
            img = Image.fromarray(
                np.pad(np.array(img), ((0,pad_h),(0,pad_w),(0,0)),
                       mode="constant", constant_values=0).astype(np.uint8)
            )
            mask = np.pad(mask, ((0,pad_h),(0,pad_w)),
                          mode="constant", constant_values=self.ignore_index)
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
            img = img.resize((int(round(img.width*s)), int(round(img.height*s))), RESAMPLE_BILINEAR)
            mask_pil = Image.fromarray(mask.astype(np.uint16), mode="I;16")
            mask_pil = mask_pil.resize(
                (int(round(mask.shape[1]*s)), int(round(mask.shape[0]*s))),
                RESAMPLE_NEAREST
            )
            mask = np.array(mask_pil, dtype=np.int64)
        th = tw = self.crop_size
        i = max(0, (img.height - th)//2)
        j = max(0, (img.width - tw)//2)
        img  = img.crop((j, i, j+tw, i+th))
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
        mask_np = mask.astype(np.int64)
        return img_np, mask_np

# ---------- tf.data pipelines ----------
def make_tf_dataset(voc: MultiRootVOCDataset, batch_size: int, shuffle: bool,
                    ignore_index: int, num_parallel_calls=tf.data.AUTOTUNE):
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
    ds = ds.map(_tf_map, num_parallel_calls=num_parallel_calls)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ---------- Evaluation callback ----------
class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, num_classes: int, ignore_index: int, ckpt_path: Path):
        super().__init__()
        self.val_ds = val_ds
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.best_miou = 0.0
        self.ckpt_path = ckpt_path
        self._bce = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

    def on_epoch_end(self, epoch, logs=None):
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        correct_px, total_px = 0, 0
        bce_sum, n_batches = 0.0, 0
        for batch in self.val_ds:
            imgs, y = batch
            masks = y["sem_logits"].numpy()
            boundary_t = y["boundary_logits"].numpy()
            sem_logits, boundary_logits = self.model(imgs, training=False)
            preds = tf.argmax(sem_logits, axis=-1).numpy().astype(np.int64)
            cm += compute_confusion_matrix(preds, masks, self.num_classes, self.ignore_index)
            valid = (masks != self.ignore_index)
            correct_px += (preds == masks)[valid].sum()
            total_px   += valid.sum()
            per_px = self._bce(boundary_t, boundary_logits).numpy()
            bce_sum += per_px.mean()
            n_batches += 1
        pixacc = correct_px / max(1, total_px)
        miou, class_ious = miou_from_confmat(cm)
        val_bce = bce_sum / max(1, n_batches)
        print(f"[Eval] valPA={pixacc:.3f}  valmIoU={miou:.3f}  valBCE={val_bce:.4f}  per-class IoU={np.round(class_ious,3)}")
        if miou > self.best_miou:
            self.best_miou = miou
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.ckpt_path.as_posix())
            print(f"Saved best to {self.ckpt_path} (mIoU {miou:.3f})")

# ---------- Helpers: latest checkpoint & epoch parsing ----------
_CKPT_RE = re.compile(r"ckpt-epoch(\d{3})\.keras$")
def find_latest_checkpoint(save_dir: Path) -> Optional[Path]:
    if not save_dir.exists():
        return None
    cands = sorted(save_dir.glob("ckpt-epoch*.keras"))
    if not cands:
        return None
    # pick highest epoch
    best = None
    best_ep = -1
    for p in cands:
        m = _CKPT_RE.search(p.name)
        if m:
            ep = int(m.group(1))
            if ep > best_ep:
                best, best_ep = p, ep
    return best

def parse_epoch_from_name(p: Path) -> int:
    m = _CKPT_RE.search(p.name)
    return int(m.group(1)) if m else 0

# ---------- Main ----------
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
    p.add_argument("--num_workers", type=int, default=4, help="Parallel map calls for tf.data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic TF ops")
    p.add_argument("--save_dir", type=str, default="models")
    # NEW: resume options
    p.add_argument("--resume", type=str, default="", 
                   help='Path to .keras checkpoint OR "auto" to pick latest; leave empty to start fresh.')
    p.add_argument("--initial_epoch", type=int, default=-1,
                   help="Override initial_epoch for fit(); if <0, auto-detect from checkpoint filename when resuming.")

    # Defaults for quick run (edit to your env)
    if len(sys.argv) == 1:
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
            epochs=3,
            batch_size=4,
            crop_size=512,
            save_dir=r"D:\animal_data\models"
        )
        args = p.parse_args([])
    else:
        args = p.parse_args()
        if not args.data_roots or not args.labelmap:
            p.error("--data_roots and --labelmap are required when passing CLI arguments.")

    roots: List[str] = []
    for item in (args.data_roots or []):
        roots.extend([s for s in item.split(",") if s])
    if not roots:
        p.error("No valid data roots provided.")

    for r in roots:
        jp = Path(r) / "JPEGImages"
        sp = Path(r) / "SegmentationClass"
        ip = Path(r) / "ImageSets" / "Segmentation"
        if not (jp.exists() and sp.exists() and ip.exists()):
            p.error(f"Root missing VOC folders: {r}")

    set_seed(args.seed, enable_determinism=args.deterministic)

    names, colors = read_labelmap(Path(args.labelmap))
    num_classes = len(names)
    print(f"Classes ({num_classes}): {names}")

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

    num_calls = args.num_workers if args.num_workers and args.num_workers > 0 else tf.data.AUTOTUNE
    train_ds = make_tf_dataset(train_ds_wrap, batch_size=args.batch_size, shuffle=True,  ignore_index=255,
                               num_parallel_calls=num_calls)
    val_ds   = make_tf_dataset(val_ds_wrap,   batch_size=1,             shuffle=False, ignore_index=255,
                               num_parallel_calls=num_calls)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # === Build or resume model ===
    model = None
    initial_epoch = 0

    # 1) RESUME path selection
    resume_path: Optional[Path] = None
    if args.resume:
        if args.resume.lower() == "auto":
            resume_path = find_latest_checkpoint(save_dir)
            if resume_path is None:
                print("[Resume] No checkpoint found in save_dir; starting fresh.")
        else:
            rp = Path(args.resume)
            if rp.exists() and rp.suffix == ".keras":
                resume_path = rp
            else:
                print(f"[Resume] Provided --resume not found or not .keras: {args.resume}")

    # 2) Load or build
    if resume_path is not None:
        print(f"[Resume] Loading {resume_path}")
        # Keras 3: load_model restores architecture + weights + optimizer state if compile=True
        model = keras.saving.load_model(resume_path.as_posix(), compile=True)
        # initial_epoch choice
        if args.initial_epoch >= 0:
            initial_epoch = args.initial_epoch
        else:
            initial_epoch = parse_epoch_from_name(resume_path)
        print(f"[Resume] initial_epoch set to {initial_epoch}")
    else:
        # Fresh build
        model = make_unet_model(num_classes)
        losses = {
            "sem_logits": SparseCEIgnoreIndex(ignore_index=255, from_logits=True),
            "boundary_logits": keras.losses.BinaryCrossentropy(from_logits=True)
        }
        loss_weights = {"sem_logits": 1.0, "boundary_logits": 1.0}
        optimizer = keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    # === Callbacks ===
    # (a) Save full model each epoch with epoch-indexed filename
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=(save_dir / "ckpt-epoch{epoch:03d}.keras").as_posix(),
        monitor="val_loss",            # will exist if you pass validation_data
        mode="min",
        save_best_only=False,          # save every epoch
        save_weights_only=False,       # FULL MODEL (.keras)
        save_freq="epoch",
        verbose=1
    )
    # (b) Your eval + best-by-mIoU
    best_ckpt_path = save_dir / "unet_boundary_best.keras"
    eval_cb = EvalCallback(val_ds, num_classes=num_classes, ignore_index=255, ckpt_path=best_ckpt_path)

    # === Train / Continue ===
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=[ckpt_cb, eval_cb],
        verbose=1
    )

if __name__ == "__main__":
    main_unet()
