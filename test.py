#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import numpy as np
from PIL import Image

# --- Pillow resampling shim (for Pillow 9/10) ---
from PIL import Image as _PILImage
if hasattr(_PILImage, "Resampling"):
    RESAMPLE_NEAREST = _PILImage.Resampling.NEAREST
else:
    RESAMPLE_NEAREST = _PILImage.NEAREST

# ---------- labelmap ----------
def read_labelmap(labelmap_path: Path):
    names, colors = [], []
    for raw in Path(labelmap_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): 
            continue
        if ":" not in line:
            raise ValueError(f"Missing ':' in labelmap line: {line}")
        name, rest = line.split(":", 1)
        name = name.strip()
        rgb = rest.split(":", 1)[0]
        r, g, b = [int(c.strip()) for c in rgb.split(",")]
        names.append(name)
        colors.append((r, g, b))
    return names, colors  # order defines class index

def build_color_to_index(colors: List[Tuple[int,int,int]]) -> Dict[Tuple[int,int,int], int]:
    return {tuple(map(int, c)): i for i, c in enumerate(colors)}

# ---------- conversions ----------
def mask_rgb_to_index(mask_img: Image.Image, color_to_index: Dict[Tuple[int,int,int], int], ignore_index=255) -> np.ndarray:
    arr = np.array(mask_img.convert("RGB"), dtype=np.uint8)  # (H,W,3)
    h, w, _ = arr.shape
    flat = arr.reshape(-1, 3)
    keys = (flat[:,0].astype(np.int32) << 16) | (flat[:,1].astype(np.int32) << 8) | flat[:,2].astype(np.int32)
    lut = {(r<<16)|(g<<8)|b: idx for (r,g,b), idx in color_to_index.items()}
    out = np.full((h*w,), ignore_index, dtype=np.uint8)
    for k, idx in lut.items():
        out[keys == k] = idx
    return out.reshape(h, w)

def index_to_color(mask_idx: np.ndarray, colors: List[Tuple[int,int,int]]) -> Image.Image:
    h, w = mask_idx.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, rgb in enumerate(colors):
        out[mask_idx == i] = rgb
    return Image.fromarray(out, mode="RGB")

# ---------- helpers ----------
def gather_mask_paths(roots: List[Path], image_set: str) -> List[Path]:
    paths = []
    for root in roots:
        set_file = root / "ImageSets" / "Segmentation" / f"{image_set}.txt"
        seg_dir  = root / "SegmentationClass"
        if not set_file.exists() or not seg_dir.exists():
            continue
        ids = [s.strip() for s in set_file.read_text().splitlines() if s.strip()]
        for _id in ids:
            p = seg_dir / f"{_id}.png"
            if p.exists():
                paths.append(p)
    return paths

def unique_colors(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    flat = arr.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    return uniq  # (K,3) uint8

# ---------- round-trip check ----------
def check_roundtrip(mask_path: Path, color_to_index, palette, out_dir: Path, sample_idx: int):
    m_rgb = Image.open(mask_path).convert("RGB")
    # IMPORTANT: if you ever resize masks, use NEAREST only (avoid invented colors)
    # m_rgb = m_rgb.resize(m_rgb.size, RESAMPLE_NEAREST)

    # colors in file
    file_colors = unique_colors(m_rgb)
    file_color_set = {tuple(int(v) for v in row.tolist()) for row in file_colors}

    # colors in labelmap (allowed)
    allowed = set(color_to_index.keys())

    # unmapped colors (exist in file but not in labelmap)
    extra = sorted(list(file_color_set - allowed))

    # convert -> index -> color
    idx = mask_rgb_to_index(m_rgb, color_to_index, ignore_index=255)
    back = index_to_color(idx, palette)

    # diff image (where back != original)
    a = np.array(m_rgb, dtype=np.uint8)
    b = np.array(back,  dtype=np.uint8)
    diff = np.any(a != b, axis=-1).astype(np.uint8) * 255

    # stats
    ignore_pct = 100.0 * np.mean(idx == 255)
    uniques, counts = np.unique(idx, return_counts=True)
    hist = {int(u): int(c) for u, c in zip(uniques, counts)}

    # save a few examples
    back.save(out_dir / f"{sample_idx:03d}_recolor.png")
    Image.fromarray(diff, mode="L").save(out_dir / f"{sample_idx:03d}_diff.png")

    return {
        "path": str(mask_path),
        "extra_colors_in_file_not_in_labelmap": extra,
        "ignore_percent": ignore_pct,
        "class_histogram": hist,
        "any_roundtrip_mismatch": bool(np.any(diff)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelmap", type=str, default=r"D:\animal_data\img_segment\labelmap.txt")
    ap.add_argument("--data_roots", nargs="+", default=[
        r"D:\animal_data\data\cheetah",
        r"D:\animal_data\data\lion",
        r"D:\animal_data\data\wolf",
        r"D:\animal_data\data\tiger",
        r"D:\animal_data\data\hyena",
        r"D:\animal_data\data\fox",
    ])
    ap.add_argument("--image_set", type=str, default="train")
    ap.add_argument("--max_samples", type=int, default=24)
    ap.add_argument("--out_dir", type=str, default=r"D:\animal_data\img_segment\label_check")
    args = ap.parse_args([]) if len(sys.argv) == 1 else ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    names, colors = read_labelmap(Path(args.labelmap))
    print(f"Loaded {len(names)} classes from labelmap:", names)
    color_to_index = build_color_to_index(colors)

    roots = [Path(p) for p in args.data_roots]
    masks = gather_mask_paths(roots, args.image_set)
    if not masks:
        print("No masks found. Check roots / image_set.")
        sys.exit(1)

    print(f"Checking up to {args.max_samples} masks (found {len(masks)})...")
    report = []
    for i, mp in enumerate(masks[:args.max_samples]):
        r = check_roundtrip(mp, color_to_index, colors, out_dir, i)
        report.append(r)
        print(f"[{i+1:02d}] {mp.name} | extra_colors={len(r['extra_colors_in_file_not_in_labelmap'])} "
              f"| ignore={r['ignore_percent']:.2f}% | mismatch={r['any_roundtrip_mismatch']}")

    # Aggregate warnings
    any_extra = any(len(r["extra_colors_in_file_not_in_labelmap"]) > 0 for r in report)
    high_ignore = [r for r in report if r["ignore_percent"] > 1.0]  # tune threshold
    any_mismatch = any(r["any_roundtrip_mismatch"] for r in report)

    print("\n=== Summary ===")
    print("Any colors in masks NOT listed in labelmap?:", any_extra)
    print("Any round-trip color→index→color mismatches?:", any_mismatch)
    if high_ignore:
        worst = max(r["ignore_percent"] for r in report)
        print(f"{len(high_ignore)} / {len(report)} samples have >1% ignore pixels (worst {worst:.2f}%).")
        print("Tip: off-palette colors often come from resizing masks without NEAREST interpolation.")

    # Optional: write a quick CSV
    csv_path = out_dir / "roundtrip_report.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("path,extra_colors_count,ignore_percent,roundtrip_mismatch\n")
        for r in report:
            f.write(f"{r['path']},{len(r['extra_colors_in_file_not_in_labelmap'])},{r['ignore_percent']:.4f},{int(r['any_roundtrip_mismatch'])}\n")
    print(f"\nWrote: {csv_path}")
    print(f"Samples saved in: {out_dir}")

if __name__ == "__main__":
    main()
