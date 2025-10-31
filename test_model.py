import os, sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import numpy as np
from PIL import Image

# --- Keras 3 backend (must be set before importing keras) ---
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import tensorflow as tf
import keras

# --- Pillow resampling shim (handles Pillow 9/10 enum change) ---
from PIL import Image as _PILImage
if hasattr(_PILImage, "Resampling"):
    RESAMPLE_BILINEAR = _PILImage.Resampling.BILINEAR
    RESAMPLE_NEAREST  = _PILImage.Resampling.NEAREST
else:
    RESAMPLE_BILINEAR = _PILImage.BILINEAR
    RESAMPLE_NEAREST  = _PILImage.NEAREST

# ---------- helpers ----------
def read_labelmap(labelmap_path: Path):
    names, colors = [], []
    if not labelmap_path.exists():
        return names, colors
    for raw in Path(labelmap_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Missing colon in line: {line}")
        name, rest = line.split(":", 1)
        name = name.strip()
        color_field = rest.split(":", 1)[0]
        r,g,b = [int(c.strip()) for c in color_field.split(",")]
        names.append(name)
        colors.append((r,g,b))
    return names, colors

def colorize_index_mask(mask: np.ndarray, colors: List[Tuple[int,int,int]]) -> Image.Image:
    h,w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    if colors:
        for idx, rgb in enumerate(colors):
            out[mask == idx] = rgb
    else:
        # fallback: generate deterministic colors
        rng = np.random.default_rng(0)
        palette = rng.integers(0, 256, size=(mask.max()+1, 3), dtype=np.uint8)
        for idx in range(palette.shape[0]):
            out[mask == idx] = palette[idx]
    return Image.fromarray(out, mode="RGB")

def save_boundary_heatmap(boundary_logits: np.ndarray, path: Path):
    if boundary_logits.ndim == 4:
        prob = tf.nn.sigmoid(boundary_logits)[0,...,0].numpy()
    elif boundary_logits.ndim == 3:
        prob = tf.nn.sigmoid(boundary_logits[...,0]).numpy()
    else:
        prob = tf.nn.sigmoid(boundary_logits).numpy()
    arr = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def preprocess(img: Image.Image) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return (arr - mean) / std

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str)
    ap.add_argument("--image_path", type=str)
    ap.add_argument("--output_dir", type=str)
    ap.add_argument("--labelmap", type=str, default=r"D:\animal_data\img_segment\labelmap.txt")
    ap.add_argument("--save_boundary", action="store_true", help="also write boundary heatmap")
    # Press-Run defaults:
    if len(sys.argv) == 1:
        ap.set_defaults(
            model_path = r"D:\animal_data\models\unet_boundary_best.keras",
            image_path = r"D:\animal_data\img_segment\data\cheetah\JPEGImages\00000000_512resized.png",
            output_dir = r"D:\animal_data\img_segment",
            save_boundary = True
        )
        args = ap.parse_args([])
    else:
        args = ap.parse_args()

    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model (inference: compile=False)
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path.as_posix(), compile=False)  # Keras docs: compile flag controls recompilation. :contentReference[oaicite:1]{index=1}

    # Load image
    print(f"Reading image: {image_path}")
    img = Image.open(image_path).convert("RGB")

    exp_h = int(model.inputs[0].shape[1])
    exp_w = int(model.inputs[0].shape[2])

    if img.size != (exp_w, exp_h):
        img = img.resize((exp_w, exp_h), RESAMPLE_BILINEAR) 
        
    H, W = img.height, img.width

    # Preprocess & run
    x = preprocess(img)[None, ...]  # (1,H,W,3)
    sem_logits, boundary_logits = model(x, training=False)

    # Semantic prediction
    pred = tf.argmax(sem_logits, axis=-1)[0].numpy().astype(np.int32)  # (H,W)
    pred_idx_path = out_dir / "pred_index.png"
    Image.fromarray(pred.astype(np.uint8), mode="L").save(pred_idx_path)

    # Optional colorized prediction
    names, colors = read_labelmap(Path(args.labelmap))
    pred_color = colorize_index_mask(pred, colors)
    pred_color_path = out_dir / "pred_color.png"
    pred_color.save(pred_color_path)

    # Optional boundary heatmap
    if args.save_boundary:
        save_boundary_heatmap(boundary_logits.numpy(), out_dir / "pred_boundary.png")

    print(f"Saved:\n  {pred_idx_path}\n  {pred_color_path}")
    if args.save_boundary:
        print(f"  {out_dir / 'pred_boundary.png'}")

if __name__ == "__main__":
    main()
