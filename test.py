#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, random
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import numpy as np
from PIL import Image

# Pillow resampling shim
from PIL import Image as _PILImage
if hasattr(_PILImage, "Resampling"):
    RESAMPLE_NEAREST  = _PILImage.Resampling.NEAREST
    RESAMPLE_BILINEAR = _PILImage.Resampling.BILINEAR
else:
    RESAMPLE_NEAREST  = _PILImage.NEAREST
    RESAMPLE_BILINEAR = _PILImage.BILINEAR

def read_labelmap(labelmap_path: Path):
    names, colors = [], []
    for raw in Path(labelmap_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): 
            continue
        if ":" not in line: 
            raise ValueError(f"Bad line: {line}")
        name, rest = line.split(":", 1)
        name = name.strip()
        rgb  = rest.split(":", 1)[0]
        r,g,b = [int(c.strip()) for c in rgb.split(",")]
        names.append(name); colors.append((r,g,b))
    return names, colors

def gather_masks(roots: List[Path], image_set: str) -> List[Path]:
    out = []
    for r in roots:
        sf = r / "ImageSets" / "Segmentation" / f"{image_set}.txt"
        sd = r / "SegmentationClass"
        if not sf.exists() or not sd.exists(): 
            continue
        ids = [s.strip() for s in sf.read_text().splitlines() if s.strip()]
        out += [sd / f"{_id}.png" for _id in ids if (sd / f"{_id}.png").exists()]
    return out

def count_off_palette(mask_rgb: Image.Image, palette_set: set) -> int:
    arr = np.array(mask_rgb.convert("RGB"), dtype=np.uint8)
    flat = arr.reshape(-1,3)
    keys = (flat[:,0].astype(np.int32)<<16) | (flat[:,1].astype(np.int32)<<8) | flat[:,2].astype(np.int32)
    return int(np.sum(~np.isin(keys, list(palette_set))))

def try_scales():
    # Down, up, and non-integer scales (these are the tricky ones)
    return [(0.5,0.5), (0.75,0.75), (1.5,1.5), (2.0,2.0), (1.25,1.25)]

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
    ap.add_argument("--max_samples", type=int, default=20)
    args = ap.parse_args([]) if len(sys.argv)==1 else ap.parse_args()

    names, colors = read_labelmap(Path(args.labelmap))
    palette_set = { (r<<16)|(g<<8)|b for (r,g,b) in colors }

    masks = gather_masks([Path(p) for p in args.data_roots], args.image_set)
    if not masks:
        print("No masks found."); sys.exit(1)
    random.shuffle(masks)
    masks = masks[:args.max_samples]

    print(f"Checking {len(masks)} masks across scales with NEAREST vs BILINEAR ...")
    for mp in masks:
        m = Image.open(mp).convert("RGB")
        H,W = m.height, m.width
        for (sy,sx) in try_scales():
            new_w, new_h = max(1, int(round(W*sx))), max(1, int(round(H*sy)))
            # Resize down/up, then back to original size
            # NEAREST path
            a = m.resize((new_w, new_h), RESAMPLE_NEAREST).resize((W,H), RESAMPLE_NEAREST)
            off_nearest = count_off_palette(a, palette_set)
            # BILINEAR path
            b = m.resize((new_w, new_h), RESAMPLE_BILINEAR).resize((W,H), RESAMPLE_BILINEAR)
            off_bilinear = count_off_palette(b, palette_set)
            print(f"{mp.name} scale({sy:.2f}x{sx:.2f})  off_NEAREST={off_nearest}  off_BILINEAR={off_bilinear}")

if __name__ == "__main__":
    main()
