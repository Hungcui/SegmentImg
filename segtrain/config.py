import argparse, os, sys
from pathlib import Path
from segtrain.utils import guess_google_drive_windows

def build_argparser():
    p = argparse.ArgumentParser(
        description="Improved U-Net training with advanced features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data_roots", nargs="+", type=str, help="List of VOC roots")
    p.add_argument("--labelmap", type=str, help="Labelmap file")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="models")
    p.add_argument("--architecture", type=str, default="unet",
                   choices=["unet", "attention_unet", "unet_plusplus", "unet_backbone"])
    p.add_argument("--backbone", type=str, default="efficientnet",
                   choices=["efficientnet", "resnet"])
    p.add_argument("--backbone_name", type=str, default="EfficientNetB0")
    p.add_argument("--loss", type=str, default="ce",
                   choices=["ce", "weighted_ce", "focal", "tversky"])
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--tversky_alpha", type=float, default=0.5)
    p.add_argument("--tversky_beta", type=float, default=0.5)
    p.add_argument("--deep_supervision", action="store_true")
    p.add_argument("--use_advanced_aug", action="store_true")
    return p

def parse_args_with_defaults():
    p = build_argparser()
    known_args, unknown_args = p.parse_known_args()
    is_colab = (os.path.exists("/content") or "COLAB_GPU" in os.environ)

    if len(sys.argv) == 1 or (len(unknown_args) > 0 and not known_args.data_roots):
        if is_colab:
            labelmap_candidates = ["/content/labelmap.txt",
                                   "/content/drive/MyDrive/SegmentImg/labelmap.txt"]
            labelmap_path = "/content/labelmap.txt"
            for c in labelmap_candidates:
                if os.path.exists(c):
                    labelmap_path = c; break
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
                epochs=50, batch_size=8, crop_size=512,
                architecture="attention_unet",
                loss="focal", use_advanced_aug=True,
                save_dir="/content/drive/MyDrive/SegmentImg/models",
            )
            print("🌐 Running on Google Colab")
            print(f"📁 Labelmap: {labelmap_path}")
            print("💾 Models will be saved to: /content/drive/MyDrive/SegmentImg/models")
        else:
            drive_root = guess_google_drive_windows()
            if drive_root is not None:
                p.set_defaults(
                    data_roots=[
                        (drive_root / "SegmentImg/data/cheetah").as_posix(),
                        (drive_root / "SegmentImg/data/lion").as_posix(),
                        (drive_root / "SegmentImg/data/wolf").as_posix(),
                        (drive_root / "SegmentImg/data/tiger").as_posix(),
                        (drive_root / "SegmentImg/data/hyena").as_posix(),
                        (drive_root / "SegmentImg/data/fox").as_posix(),
                    ],
                    labelmap=(drive_root / "SegmentImg/labelmap.txt").as_posix(),
                    epochs=50, batch_size=8, crop_size=512,
                    architecture="attention_unet", loss="focal",
                    use_advanced_aug=True,
                    save_dir=(drive_root / "SegmentImg/models").as_posix(),
                )
                print(f"💻 Running on Local Machine (Google Drive detected at: {drive_root})")
            else:
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
                    epochs=5, batch_size=4, crop_size=512,
                    architecture="attention_unet", loss="focal",
                    use_advanced_aug=True, save_dir="models",
                )
                print("💻 Running on Local Machine (no Google Drive found)")
        return p.parse_args(unknown_args)
    else:
        args = known_args
        if not args.data_roots or not args.labelmap:
            p.error("--data_roots and --labelmap are required")
        return args

def check_paths(roots, labelmap, save_dir, is_colab: bool):
    from pathlib import Path
    print("\n" + "="*60)
    print("CHECKING DATA PATHS")
    print("="*60)
    missing = []
    for r in roots:
        jp = Path(r) / "JPEGImages"
        sp = Path(r) / "SegmentationClass"
        ip = Path(r) / "ImageSets" / "Segmentation"
        if jp.exists() and sp.exists() and ip.exists():
            n_images = len(list(jp.glob("*"))) if jp.exists() else 0
            print(f"✅ {Path(r).name}: {n_images} images")
        else:
            print(f"❌ {Path(r).name}: Missing VOC folders")
            missing.append(r)
    if missing:
        print(f"\n  Warning: {len(missing)} dataset(s) not found!")
        if is_colab:
            print("💡 Tip: Mount Drive and place data under /content/drive/MyDrive/SegmentImg/data/")
        print("\nContinue anyway? (y/n): ", end="")
        if not is_colab:
            if input().strip().lower() != 'y':
                raise SystemExit("Please check your data paths and try again")

    lp = Path(labelmap)
    if not lp.exists():
        print(f"\n❌ Labelmap not found: {labelmap}")
        if is_colab:
            print("💡 Tip: Upload labelmap.txt to /content/ or /content/drive/MyDrive/SegmentImg/")
        raise SystemExit(f"Labelmap file not found: {labelmap}")
    else: print(f"✅ Labelmap: {labelmap}")

    sd = Path(save_dir); sd.mkdir(parents=True, exist_ok=True)
    print(f"💾 Save directory: {save_dir}")
    print("="*60 + "\n")
