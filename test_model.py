import argparse
from pathlib import Path
from typing import List, Tuple
import shutil

import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Label map
# -----------------------------
def read_labelmap(labelmap_path: Path):
    """Read 'label: R,G,B' per line; ignores blanks/#comments."""
    names, colors = [], []
    text = Path(labelmap_path).read_text(encoding="utf-8").splitlines()
    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Check labelmap line (no colon): {line}")
        name, rest = line.split(":", 1)
        name = name.strip()
        comps = rest.split(":", 1)[0].split(",")
        if len(comps) != 3:
            raise ValueError(f"RGB must have 3 components: {line}")
        r, g, b = [int(c.strip()) for c in comps]
        names.append(name)
        colors.append((r, g, b))
    return names, colors

# -----------------------------
# I/O helpers
# -----------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def center_crop_or_resize(img: Image.Image, size: int) -> Image.Image:
    """Resize short side to >= size, then center-crop to (size,size)."""
    w, h = img.size
    short = min(w, h)
    if short < size:
        s = size / short
        img = img.resize((int(round(w*s)), int(round(h*s))), Image.BILINEAR)
    w, h = img.size
    i = max(0, (h - size)//2)
    j = max(0, (w - size)//2)
    return img.crop((j, i, j+size, i+size))

def to_tensor(img_rgb: Image.Image) -> np.ndarray:
    arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr  # (H,W,3)

def save_indexed_palette_mask(save_path: Path, pred_idx: np.ndarray, colors: List[Tuple[int,int,int]]):
    """
    Save an indexed (mode 'P') PNG where each pixel value is the class id and
    the palette rows are exactly the RGB colors from `colors`.
    (Palette images support up to 256 colors.) :contentReference[oaicite:1]{index=1}
    """
    maskP = Image.fromarray(pred_idx.astype(np.uint8), mode="P")
    palette = []
    for (r, g, b) in colors:
        palette.extend([int(r), int(g), int(b)])
    palette += [0, 0, 0] * (256 - len(colors))
    maskP.putpalette(palette)
    maskP.save(save_path)

# -----------------------------
# Inference (ONLY the essentials)
# -----------------------------
def run_inference(
    model_path: Path,
    image_path: Path,
    labelmap_path: Path,
    out_dir: Path,
    crop_size: int = 512,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load model
    model = tf.keras.models.load_model(model_path.as_posix(), compile=False)

    # 2) Classes/colors
    names, colors = read_labelmap(labelmap_path)

    # 3) Read + preprocess image
    orig = load_image_rgb(image_path)
    proc = center_crop_or_resize(orig, crop_size)
    x = to_tensor(proc)[None, ...]  # (1,H,W,3)

    # 4) Forward pass -> softmax -> argmax
    sem_logits = model(x, training=False)[0]  # first head: (1,H,W,C) or dict["sem_logits"]
    if isinstance(sem_logits, dict):  # if model returns dict
        sem_logits = sem_logits["sem_logits"]
    sem_prob = tf.nn.softmax(sem_logits, axis=-1).numpy()[0]  # (H,W,C). Sum along C=1. :contentReference[oaicite:2]{index=2}
    sem_pred = np.argmax(sem_prob, axis=-1).astype(np.uint8)  # (H,W)

    # 5) Resize mask back to original image size (nearest neighbor) and save as palettized PNG. :contentReference[oaicite:3]{index=3}
    sem_pred_up = np.array(
        Image.fromarray(sem_pred, mode="L").resize(orig.size, Image.NEAREST)
    )
    save_indexed_palette_mask(out_dir / "segmented.png", sem_pred_up, colors)

    # 6) Classification list (unique class ids present, excluding background=0). :contentReference[oaicite:4]{index=4}
    present_ids = np.unique(sem_pred_up)
    present_ids = [int(i) for i in present_ids if 0 < i < len(names)]
    with open(out_dir / "classes_present.txt", "w", encoding="utf-8") as f:
        for cid in present_ids:
            f.write(f"{cid}\t{names[cid]}\n")

    # 7) Copy original image into out_dir (no metadata). :contentReference[oaicite:5]{index=5}
    dst = out_dir / f"original{Path(image_path).suffix}"
    shutil.copyfile(image_path, dst.as_posix())

def main():
    ap = argparse.ArgumentParser(description="Minimal test: save palettized segmentation + classes + original image")
    ap.add_argument("--model", type=str, default=r"D:\animal_data\models\unet_boundary_best.keras")
    ap.add_argument("--image", type=str, default=r"D:\animal_data\test_data\cheetah\00000191_512resized.png")
    ap.add_argument("--labelmap", type=str, default=r"D:\animal_data\img_segment\labelmap.txt")
    ap.add_argument("--out_dir", type=str, default=r"D:\animal_data\test_outputs")
    ap.add_argument("--crop_size", type=int, default=512)
    args = ap.parse_args()

    run_inference(
        model_path=Path(args.model),
        image_path=Path(args.image),
        labelmap_path=Path(args.labelmap),
        out_dir=Path(args.out_dir),
        crop_size=args.crop_size,
    )

if __name__ == "__main__":
    main()
