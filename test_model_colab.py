"""
Test Model trên Google Colab
Sử dụng: Upload file này lên Colab và chạy với arguments hoặc không có arguments (dùng defaults)
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import argparse

# Phát hiện Colab
is_colab = os.path.exists("/content") or "COLAB_GPU" in os.environ

def read_labelmap(labelmap_path: Path):
    """Đọc labelmap file"""
    names, colors = [], []
    if not labelmap_path.exists():
        return names, colors
    for raw in Path(labelmap_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        name = name.strip()
        color_field = rest.split(":", 1)[0]
        r, g, b = [int(c.strip()) for c in color_field.split(",")]
        names.append(name)
        colors.append((r, g, b))
    return names, colors

def colorize_index_mask(mask: np.ndarray, colors):
    """Chuyển mask index sang màu RGB"""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if colors:
        for idx, rgb in enumerate(colors):
            out[mask == idx] = rgb
    return Image.fromarray(out, mode="RGB")

def preprocess(img: Image.Image) -> np.ndarray:
    """Chuẩn hóa ảnh theo ImageNet"""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return (arr - mean) / std

def main():
    parser = argparse.ArgumentParser(description="Test model trên Colab")
    parser.add_argument("--model_path", type=str, help="Đường dẫn model file (.keras)")
    parser.add_argument("--image_path", type=str, help="Đường dẫn ảnh test")
    parser.add_argument("--output_dir", type=str, help="Thư mục lưu kết quả")
    parser.add_argument("--labelmap", type=str, help="Đường dẫn labelmap.txt")
    parser.add_argument("--save_boundary", action="store_true", help="Lưu boundary heatmap")
    
    # Defaults cho Colab
    if len(sys.argv) == 1 or (is_colab and not any(['--model_path' in s or '--image_path' in s for s in sys.argv])):
        if is_colab:
            # Tự động tìm model và labelmap
            model_candidates = [
                "/content/drive/MyDrive/SegmentImg/models/attention_unet_focal_best.keras",
                "/content/drive/MyDrive/SegmentImg/models/unet_boundary_best.keras",
                "/content/models/attention_unet_focal_best.keras",
            ]
            model_path = None
            for candidate in model_candidates:
                if Path(candidate).exists():
                    model_path = candidate
                    break
            
            labelmap_candidates = [
                "/content/labelmap.txt",
                "/content/drive/MyDrive/SegmentImg/labelmap.txt",
            ]
            labelmap_path = "/content/labelmap.txt"
            for candidate in labelmap_candidates:
                if Path(candidate).exists():
                    labelmap_path = candidate
                    break
            
            # Tìm ảnh test trong data
            image_candidates = [
                "/content/drive/MyDrive/SegmentImg/data/cheetah/JPEGImages/00000000_512resized.png",
                "/content/drive/MyDrive/SegmentImg/data/lion/JPEGImages/00000000_512resized.png",
            ]
            image_path = None
            for candidate in image_candidates:
                if Path(candidate).exists():
                    image_path = candidate
                    break
            
            parser.set_defaults(
                model_path=model_path or "/content/drive/MyDrive/SegmentImg/models/attention_unet_focal_best.keras",
                image_path=image_path or "/content/test_image.jpg",
                output_dir="/content/drive/MyDrive/SegmentImg/test_results",
                labelmap=labelmap_path,
                save_boundary=True
            )
        else:
            # Local defaults
            parser.set_defaults(
                model_path=r"D:\animal_data\models\unet_boundary_best.keras",
                image_path=r"D:\animal_data\data\cheetah\JPEGImages\00000000_512resized.png",
                output_dir=r"D:\animal_data\test_results",
                labelmap=r"D:\animal_data\img_segment\labelmap.txt",
                save_boundary=True
            )
    
    args = parser.parse_args()
    
    # Kiểm tra paths
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    labelmap_path = Path(args.labelmap)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("TEST MODEL")
    print("="*60)
    print(f"🌐 Running on: {'Google Colab' if is_colab else 'Local Machine'}")
    print(f"📦 Model: {model_path}")
    print(f"🖼️  Image: {image_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🏷️  Labelmap: {labelmap_path}")
    print("="*60)
    
    # Kiểm tra files tồn tại
    if not model_path.exists():
        print(f"\n❌ Model không tồn tại: {model_path}")
        if is_colab:
            print("💡 Tip: Upload model vào Drive hoặc chỉ định đường dẫn đúng")
        sys.exit(1)
    
    if not image_path.exists():
        print(f"\n❌ Ảnh không tồn tại: {image_path}")
        if is_colab:
            print("💡 Tip: Upload ảnh test hoặc chỉ định đường dẫn đúng")
        sys.exit(1)
    
    if not labelmap_path.exists():
        print(f"\n❌ Labelmap không tồn tại: {labelmap_path}")
        sys.exit(1)
    
    # Tạo output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n📦 Loading model...")
    try:
        model = keras.models.load_model(model_path.as_posix(), compile=False)
        print(f"✅ Model loaded! Input shape: {model.input_shape}")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        sys.exit(1)
    
    # Load labelmap
    names, colors = read_labelmap(labelmap_path)
    num_classes = len(names)
    print(f"✅ Labelmap loaded! Classes: {names}")
    
    # Load và preprocess image
    print(f"\n🖼️  Loading image...")
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size
    print(f"   Original size: {orig_size}")
    
    # Resize nếu cần (model có thể expect fixed size)
    in_shape = model.input_shape
    if len(in_shape) == 4 and in_shape[1] is not None and in_shape[2] is not None:
        exp_h, exp_w = in_shape[1], in_shape[2]
        if img.size != (exp_w, exp_h):
            img = img.resize((exp_w, exp_h), Image.BILINEAR)
            print(f"   Resized to: {exp_w}x{exp_h}")
    
    # Preprocess
    x = preprocess(img)[None, ...]  # (1, H, W, 3)
    
    # Inference
    print(f"\n🔮 Running inference...")
    outputs = model(x, training=False)
    
    # Parse outputs
    if isinstance(outputs, list):
        sem_logits = outputs[0]
        boundary_logits = outputs[1] if len(outputs) > 1 else None
    elif isinstance(outputs, dict):
        sem_logits = outputs.get("sem_logits")
        boundary_logits = outputs.get("boundary_logits")
    else:
        sem_logits = outputs
        boundary_logits = None
    
    # Get prediction
    pred = tf.argmax(sem_logits, axis=-1)[0].numpy().astype(np.int32)
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # 1. Index mask (PNG)
    Image.fromarray(pred.astype(np.uint8), mode="L").save(output_dir / "pred_index.png")
    
    # 2. Colorized mask
    pred_color = colorize_index_mask(pred, colors)
    pred_color.save(output_dir / "pred_color.png")
    
    # 3. Boundary heatmap (nếu có)
    if boundary_logits is not None and args.save_boundary:
        if boundary_logits.ndim == 4:
            boundary_prob = tf.nn.sigmoid(boundary_logits)[0, ..., 0].numpy()
        else:
            boundary_prob = tf.nn.sigmoid(boundary_logits[..., 0]).numpy()
        boundary_img = Image.fromarray((boundary_prob * 255).astype(np.uint8), mode="L")
        boundary_img.save(output_dir / "pred_boundary.png")
        print(f"   ✅ Saved boundary heatmap")
    
    # 4. Overlay trên ảnh gốc (resize về kích thước gốc)
    if orig_size != img.size:
        pred_resized = Image.fromarray(pred.astype(np.uint8), mode="L").resize(orig_size, Image.NEAREST)
        pred_color_resized = colorize_index_mask(np.array(pred_resized), colors)
        
        # Blend với ảnh gốc
        img_orig = Image.open(image_path).convert("RGB")
        overlay = Image.blend(img_orig, pred_color_resized, 0.5)
        overlay.save(output_dir / "pred_overlay.png")
        print(f"   ✅ Saved overlay")
    
    print(f"\n✅ Test completed!")
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   - pred_index.png (grayscale mask)")
    print(f"   - pred_color.png (colorized mask)")
    if boundary_logits is not None and args.save_boundary:
        print(f"   - pred_boundary.png (boundary heatmap)")
    if orig_size != img.size:
        print(f"   - pred_overlay.png (overlay on original image)")
    
    # Hiển thị prediction stats
    unique_classes, counts = np.unique(pred, return_counts=True)
    print(f"\n📊 Prediction statistics:")
    for cls_id, count in zip(unique_classes, counts):
        if cls_id < len(names):
            print(f"   {names[cls_id]}: {count} pixels ({count/pred.size*100:.1f}%)")

if __name__ == "__main__":
    main()

