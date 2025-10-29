#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VOC-style dataset sanity checks for your multi-root setup.

Usage (CLI):
  python dataset_sanity.py --data_roots D:\animal_data\img_segment\data\cheetah \
                           D:\animal_data\img_segment\data\lion \
                           --labelmap D:\animal_data\img_segment\labelmap.txt

Options:
  --check_all           Check every ID in each set (default: only a sample).
  --max_per_set N       Cap how many IDs to check per set (default: 300).
  --sets train val      Which sets to check (default: train val).
  --json OUT.json       Also save the report as JSON.

Import (from your training script):
  from dataset_sanity import run_sanity_checks, print_sanity_report
  report, ok = run_sanity_checks(roots, labelmap_path, image_sets=("train","val"), max_check_per_set=300)
  if not ok:
      raise SystemExit("Sanity check failed. Fix issues and rerun.")

python dataset_sanity.py --data_roots D:\animal_data\img_segment\data\cheetah D:\animal_data\img_segment\data\lion --labelmap D:\animal_data\img_segment\labelmap.txt --max_per_set 300
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------
# Config / small helpers
# -----------------------

IMG_EXTS = (".jpg", ".jpeg", ".png")

def _first_existing_with_ext(base: Path, exts=IMG_EXTS) -> Path | None:
    for ext in exts:
        p = base.with_suffix(ext)
        if p.exists():
            return p
    return None

# -----------------------
# Labelmap (duplicate of your logic to keep this file standalone)
# -----------------------

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

        name, rest = line.split(":", 1)
        name = name.strip()

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

def sanity_check_labelmap(labelmap_path: Path) -> Dict:
    rep = {"labelmap_path": str(labelmap_path), "errors": [], "warnings": []}
    if not labelmap_path.exists():
        rep["errors"].append(f"Labelmap not found: {labelmap_path}")
        return rep

    try:
        names, colors = read_labelmap(labelmap_path)
        rep["num_classes"] = len(names)
        rep["class_names"] = names
    except Exception as e:
        rep["errors"].append(f"Parsing labelmap failed: {e}")
        return rep

    if len(names) == 0:
        rep["errors"].append("Labelmap produced zero classes.")

    # Duplicate checks
    if len(set(names)) != len(names):
        seen = {}
        dups = []
        for n in names:
            seen[n] = seen.get(n, 0) + 1
            if seen[n] == 2:
                dups.append(n)
        rep["errors"].append(f"Duplicate class names: {dups}")

    if len(set(colors)) != len(colors):
        seen = {}
        dups = []
        for c in colors:
            seen[c] = seen.get(c, 0) + 1
            if seen[c] == 2:
                dups.append(c)
        rep["warnings"].append(f"Duplicate RGB rows: {dups}")

    for c in colors:
        if any((v < 0 or v > 255) for v in c):
            rep["errors"].append(f"Out-of-range RGB value: {c}")
            break

    return rep

# -----------------------
# Root checks
# -----------------------

def sanity_check_roots(
    roots: List[str | Path],
    image_sets: Tuple[str, ...] = ("train", "val"),
    max_check_per_set: int | None = 300,
) -> Dict:
    report = {"roots": [], "errors": [], "warnings": []}

    for r in roots:
        root = Path(r)
        rrep = {
            "root": str(root),
            "exists": root.exists(),
            "errors": [],
            "warnings": [],
            "sets": {},
        }

        if not root.exists():
            rrep["errors"].append("Root does not exist.")
            report["roots"].append(rrep)
            continue

        jp = root / "JPEGImages"
        sp = root / "SegmentationClass"
        ip = root / "ImageSets" / "Segmentation"

        for pth, name in [(jp, "JPEGImages"), (sp, "SegmentationClass"), (ip, "ImageSets/Segmentation")]:
            if not pth.exists():
                rrep["errors"].append(f"Missing folder: {name} -> {pth}")

        for set_name in image_sets:
            set_file = ip / f"{set_name}.txt"
            srep = {
                "set": set_name,
                "set_file": str(set_file),
                "exists": set_file.exists(),
                "num_ids": 0,
                "checked": 0,
                "missing_images": [],
                "missing_masks": [],
            }

            if not set_file.exists():
                srep["error"] = f"Set file missing: {set_file}"
                rrep["errors"].append(srep["error"])
                rrep["sets"][set_name] = srep
                continue

            ids = [s.strip() for s in set_file.read_text().splitlines() if s.strip()]
            srep["num_ids"] = len(ids)

            ids_to_check = ids if (max_check_per_set is None or len(ids) <= max_check_per_set) else ids[:max_check_per_set]

            for img_id in ids_to_check:
                img_base = jp / img_id
                img_path = _first_existing_with_ext(img_base)
                if img_path is None:
                    srep["missing_images"].append(str(img_base) + ".*")

                mask_path = sp / f"{img_id}.png"
                if not mask_path.exists():
                    srep["missing_masks"].append(str(mask_path))

            srep["checked"] = len(ids_to_check)

            # Trim very long lists to keep console readable
            if len(srep["missing_images"]) > 10:
                srep["missing_images"] = srep["missing_images"][:10] + ["..."]
            if len(srep["missing_masks"]) > 10:
                srep["missing_masks"] = srep["missing_masks"][:10] + ["..."]

            # Bubble up summaries
            if srep["missing_images"]:
                rrep["errors"].append(f"[{set_name}] Missing images for some IDs (showing up to 10).")
            if srep["missing_masks"]:
                rrep["errors"].append(f"[{set_name}] Missing masks for some IDs (showing up to 10).")

            rrep["sets"][set_name] = srep

        report["roots"].append(rrep)

    return report

# -----------------------
# Pretty-print & driver
# -----------------------

def print_sanity_report(full_report: Dict) -> bool:
    """Pretty-print; return True if no errors found."""
    ok = True
    def _log(m): print(m)

    lm = full_report.get("labelmap")
    if lm:
        _log(f"[Labelmap] path={lm.get('labelmap_path')}  classes={lm.get('num_classes','?')}")
        for e in lm.get("errors", []):
            _log(f"  ERROR: {e}")
            ok = False
        for w in lm.get("warnings", []):
            _log(f"  warn : {w}")

    for rr in full_report.get("roots", []):
        _log(f"[Root] {rr['root']}")
        if not rr["exists"]:
            _log("  ERROR: root does not exist")
            ok = False
        for e in rr.get("errors", []):
            _log(f"  ERROR: {e}")
            ok = False
        for w in rr.get("warnings", []):
            _log(f"  warn : {w}")
        for set_name, srep in rr.get("sets", {}).items():
            _log(f"  - Set '{set_name}': ids={srep['num_ids']} checked={srep['checked']}")
            if srep.get("missing_images"):
                _log(f"    missing images ({len(srep['missing_images'])} shown):")
                for p in srep["missing_images"]:
                    _log(f"      {p}")
            if srep.get("missing_masks"):
                _log(f"    missing masks ({len(srep['missing_masks'])} shown):")
                for p in srep["missing_masks"]:
                    _log(f"      {p}")
    return ok

def run_sanity_checks(
    roots: List[str | Path],
    labelmap_path: str | Path,
    image_sets: Tuple[str, ...] = ("train","val"),
    max_check_per_set: int | None = 300,
):
    lm_report = sanity_check_labelmap(Path(labelmap_path))
    roots_report = sanity_check_roots(roots, image_sets=image_sets, max_check_per_set=max_check_per_set)
    full = {"labelmap": lm_report, "roots": roots_report["roots"]}
    ok = print_sanity_report(full)
    return full, ok

def _parse_args():
    pa = argparse.ArgumentParser(
        description="Sanity-check VOC-style datasets (multi-root).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    pa.add_argument("--data_roots", nargs="+", required=True, help="List of VOC roots.")
    pa.add_argument("--labelmap", required=True, help="Path to labelmap.txt.")
    pa.add_argument("--sets", nargs="+", default=["train","val"], help="Which sets to check.")
    pa.add_argument("--check_all", action="store_true", help="Check all IDs in each set.")
    pa.add_argument("--max_per_set", type=int, default=300, help="Cap per-set checks (ignored if --check_all).")
    pa.add_argument("--json", type=str, default=None, help="Optional path to save report as JSON.")
    return pa.parse_args()

def main():
    args = _parse_args()
    # Normalize roots (also allow comma-separated entries)
    roots: List[str] = []
    for item in (args.data_roots or []):
        roots.extend([s for s in str(item).split(",") if s])

    max_check = None if args.check_all else args.max_per_set

    report, ok = run_sanity_checks(
        roots=roots,
        labelmap_path=args.labelmap,
        image_sets=tuple(args.sets),
        max_check_per_set=max_check
    )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to {args.json}")

    if not ok:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
