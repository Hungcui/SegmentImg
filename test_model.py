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
    """Map nhãn 0..C-1 -> RGB rời rạc để xem trực quan (không dùng cmap liên tục)."""
    h,w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    if colors:
        for idx, rgb in enumerate(colors):
            out[mask == idx] = rgb
    else:
        # fallback: deterministic palette
        k = int(mask.max()) + 1
        rng = np.random.default_rng(0)
        palette = rng.integers(0, 256, size=(k, 3), dtype=np.uint8)
        for idx in range(k):
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
    # Chuẩn ImageNet: mean/std phổ biến khi train backbone kiểu ImageNet
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

    # Load model for inference (no need to recompile)
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path.as_posix(), compile=False)  # inference only

    # Read image
    print(f"Reading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Determine expected input size safely (supports dynamic None)
    in_shape = model.input_shape  # e.g. (None, H, W, 3) or (None, None, None, 3)
    exp_h = in_shape[1]
    exp_w = in_shape[2]

    # If model expects fixed size, resize the RGB image with bilinear.
    if (exp_h is not None) and (exp_w is not None) and (img.size != (exp_w, exp_h)):
        img = img.resize((exp_w, exp_h), RESAMPLE_BILINEAR)

    H, W = img.height, img.width

    # Preprocess & forward
    x = preprocess(img)[None, ...]  # (1,H,W,3)
    sem_logits, boundary_logits = model(x, training=False)

    # Semantic prediction (indices 0..C-1)
    pred = tf.argmax(sem_logits, axis=-1)[0].numpy().astype(np.int32)  # (H,W)

    # ---- Save index mask in 3 formats ----
    # 1) NPY (lossless, giữ nguyên dtype)
    np.save(out_dir / "pred_index.npy", pred)
    # 2) PNG 8-bit để xem nhanh (OK nếu lớp ≤255)
    Image.fromarray(pred.astype(np.uint8), mode="L").save(out_dir / "pred_index.png")
    # 3) PNG 16-bit để an toàn khi lớp có thể >255
    Image.fromarray(pred.astype(np.uint16), mode="I;16").save(out_dir / "pred_index_u16.png")

    # ---- Optional colorized prediction for visualization ----
    names, colors = read_labelmap(Path(args.labelmap))
    pred_color = colorize_index_mask(pred, colors)
    pred_color.save(out_dir / "pred_color.png")

    # ---- Optional boundary heatmap ----
    if args.save_boundary:
        save_boundary_heatmap(boundary_logits.numpy(), out_dir / "pred_boundary.png")

    print("Saved:")
    print(f"  {out_dir/'pred_index.npy'}")
    print(f"  {out_dir/'pred_index.png'}")
    print(f"  {out_dir/'pred_index_u16.png'}")
    print(f"  {out_dir/'pred_color.png'}")
    if args.save_boundary:
        print(f"  {out_dir/'pred_boundary.png'}")

    # NOTE:
    # - Nếu sau này cần resize mask dự đoán về size gốc để overlay,
    #   hãy dùng RESAMPLE_NEAREST cho mask để không “pha” nhãn rời rạc.
    #   vd: Image.fromarray(pred.astype(np.uint16), 'I;16').resize((orig_w, orig_h), RESAMPLE_NEAREST)

if __name__ == "__main__":
    main()
