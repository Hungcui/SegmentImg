# Inference Script với Instance Segmentation
# Usage: python inference_instance.py --model_path <path> --image_path <path> --output_path <path> --labelmap <path>

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import keras
from scipy.ndimage import label, binary_closing, binary_dilation
from model_train_v3_improved import (
    read_labelmap, instances_from_sem_and_boundary
)

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess image for model input"""
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    return img_np

def main():
    parser = argparse.ArgumentParser(description="Inference with Instance Segmentation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--labelmap", type=str, required=True, help="Path to labelmap file")
    parser.add_argument("--crop_size", type=int, default=512, help="Input size for model")
    parser.add_argument("--use_instance", action="store_true", default=True, help="Use instance segmentation")
    parser.add_argument("--thing_class_ids", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], 
                       help="Class IDs that are 'things' (instances), exclude background=0")
    parser.add_argument("--sem_thresh", type=float, default=0.5, help="Semantic probability threshold")
    parser.add_argument("--boundary_thresh", type=float, default=0.5, help="Boundary probability threshold")
    parser.add_argument("--min_instance_size", type=int, default=None, help="Minimum instance size in pixels (auto if None)")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model = keras.models.load_model(args.model_path, compile=False)
    
    print(f"Loading labelmap from {args.labelmap}...")
    names, colors = read_labelmap(Path(args.labelmap))
    num_classes = len(names)
    print(f"Classes ({num_classes}): {names}")
    
    print(f"Loading image: {args.image_path}")
    img = Image.open(args.image_path).convert("RGB")
    original_w, original_h = img.size  # (W, H)
    original_size = (original_w, original_h)  # cv2.resize expects (width, height)
    
    # Preprocess
    img_np = preprocess_image(img)
    
    # Resize/pad to crop_size
    h, w = img_np.shape[:2]
    scale = min(args.crop_size / h, args.crop_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to crop_size
    pad_h = args.crop_size - new_h
    pad_w = args.crop_size - new_w
    img_padded = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    
    # Inference
    print("Running inference...")
    img_tf = tf.expand_dims(tf.constant(img_padded, dtype=tf.float32), 0)
    outputs = model(img_tf, training=False)
    
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
    
    # Remove padding
    sem_logits = sem_logits[0, :new_h, :new_w]  # (H, W, C)
    if boundary_logits is not None:
        if boundary_logits.ndim == 4:
            boundary_logits = boundary_logits[0, :new_h, :new_w, 0]  # (H, W)
        else:
            boundary_logits = boundary_logits[0, :new_h, :new_w]  # (H, W)
    
    # Resize back to original size
    sem_probs = tf.nn.softmax(sem_logits, axis=-1).numpy()  # (H, W, C)
    
    # Use PIL for resizing (more reliable than cv2.resize with float32)
    sem_probs_resized = np.zeros((original_h, original_w, num_classes), dtype=np.float32)
    for c in range(num_classes):
        prob_channel = sem_probs[:, :, c]
        # Convert to PIL Image, resize, then back to numpy
        prob_img = Image.fromarray((prob_channel * 255).astype(np.uint8), mode='L')
        prob_img_resized = prob_img.resize((original_w, original_h), Image.BILINEAR)
        prob_resized = np.asarray(prob_img_resized, dtype=np.float32) / 255.0
        sem_probs_resized[:, :, c] = prob_resized
    
    # Semantic prediction
    mask_pred = np.argmax(sem_probs_resized, axis=-1).astype(np.uint8)
    
    # Instance segmentation (if enabled and boundary available)
    if args.use_instance:
        if boundary_logits is None:
            print("⚠️  Warning: Boundary logits not available. Cannot perform instance segmentation.")
            print("   Instance segmentation requires a model with boundary output.")
        else:
            print("Running instance segmentation...")
            
            # Convert to numpy first
            boundary_np = boundary_logits.numpy() if isinstance(boundary_logits, tf.Tensor) else boundary_logits
            boundary_np = boundary_np.astype(np.float32)
            
            print(f"   Boundary logits shape: {boundary_np.shape}")
            print(f"   Boundary logits range: [{boundary_np.min():.3f}, {boundary_np.max():.3f}]")
            
            # Apply sigmoid to get probability if it's logits
            boundary_prob = tf.nn.sigmoid(boundary_np).numpy() if isinstance(boundary_logits, tf.Tensor) else (1 / (1 + np.exp(-boundary_np)))
            print(f"   Boundary prob range: [{boundary_prob.min():.3f}, {boundary_prob.max():.3f}]")
            
            # Use PIL for resizing
            boundary_img = Image.fromarray((boundary_prob * 255).astype(np.uint8), mode='L')
            boundary_img_resized = boundary_img.resize((original_w, original_h), Image.BILINEAR)
            boundary_resized = np.asarray(boundary_img_resized, dtype=np.float32) / 255.0
            boundary_logits_resized = np.expand_dims(boundary_resized, axis=-1)  # (H, W, 1)
            
            # Convert to logits format for function
            sem_logits_resized = np.log(sem_probs_resized + 1e-8)  # Convert back to logits
            sem_logits_resized = np.transpose(sem_logits_resized, (2, 0, 1))  # (C, H, W)
            sem_logits_resized = np.expand_dims(sem_logits_resized, axis=0)  # (1, C, H, W)
            
            print(f"   Thing class IDs: {args.thing_class_ids}")
            print(f"   Semantic threshold: {args.sem_thresh}, Boundary threshold: {args.boundary_thresh}")
            
            # Check semantic prediction for the target class
            sem_pred_class = np.argmax(sem_probs_resized, axis=-1)
            for cid in args.thing_class_ids:
                pixels_in_class = (sem_pred_class == cid).sum()
                print(f"   Class {cid} ({names[cid]}): {pixels_in_class} pixels detected")
            
            # Simplified approach: Use semantic mask + connected components
            # This ensures each animal = 1 connected component = 1 solid color
            # No watershed = no fragmentation, keeps all parts
            print(f"   Using semantic mask + connected components (no watershed)")
            print(f"   This ensures each animal = 1 instance = 1 solid color")
            
            H, W = sem_pred_class.shape
            instance_map = np.zeros((H, W), dtype=np.int32)
            cur_label = 1
            
            # Process each class separately
            for cid in args.thing_class_ids:
                if cid < 0 or cid >= sem_probs_resized.shape[2]:
                    continue
                
                # Get mask for this class from semantic prediction
                class_mask = (sem_pred_class == cid)
                if class_mask.sum() == 0:
                    continue
                
                # Use morphological closing to connect nearby segments
                # This merges segments that are close together (parts of same animal)
                # Structure size: larger = merge more aggressively
                closing_size = 20  # Larger to merge more aggressively
                closed_mask = binary_closing(class_mask, structure=np.ones((closing_size, closing_size)))
                
                # Use connected components - each connected region = 1 instance
                # This ensures each animal (even if fragmented initially) becomes 1 instance
                labeled, num_features = label(closed_mask)
                
                # Assign new instance IDs
                for label_id in range(1, num_features + 1):
                    # Only assign pixels that were in original class_mask (don't add pixels from closing)
                    instance_pixels = (labeled == label_id) & class_mask
                    if instance_pixels.sum() > 0:
                        instance_map[instance_pixels] = cur_label
                        cur_label += 1
                
                print(f"   Class {cid} ({names[cid]}): {num_features} instances (each = 1 solid color)")
            
            # Filter small instances (noise removal) - after merging
            unique_ids = np.unique(instance_map)
            num_instances_before = len(unique_ids[unique_ids > 0])
            
            # Remove instances smaller than threshold
            if args.min_instance_size is None:
                # Auto: at least 0.2% of image, but minimum 200 pixels
                min_instance_size = max(200, (instance_map.shape[0] * instance_map.shape[1]) // 500)
            else:
                min_instance_size = args.min_instance_size
            filtered_map = np.zeros_like(instance_map)
            valid_labels = []
            
            for inst_id in unique_ids:
                if inst_id == 0:
                    continue
                mask = (instance_map == inst_id)
                size = mask.sum()
                if size >= min_instance_size:
                    valid_labels.append(inst_id)
                    filtered_map[mask] = inst_id
            
            # Re-label to make IDs consecutive
            instance_map_clean = np.zeros_like(filtered_map)
            new_id = 1
            for old_id in valid_labels:
                instance_map_clean[filtered_map == old_id] = new_id
                new_id += 1
            
            instance_map = instance_map_clean
            unique_ids = np.unique(instance_map)
            num_instances = len(unique_ids[unique_ids > 0])
            
            print(f"   Found {num_instances_before} instances before filtering")
            print(f"   Found {num_instances} instances after filtering (min size: {min_instance_size} pixels)")
            
            if num_instances == 0:
                print("   ⚠️  No instances detected. Try:")
                print("      - Lowering --sem_thresh (e.g., 0.2)")
                print("      - Lowering --boundary_thresh (e.g., 0.5)")
                print("      - Checking if thing_class_ids match detected classes")
            
            # Create RGB instance mask (giống format SegmentationObject)
            # Mỗi instance có một màu RGB riêng biệt, background = đen (0,0,0)
            instance_rgb = np.zeros((instance_map.shape[0], instance_map.shape[1], 3), dtype=np.uint8)
            
            # Generate distinct colors for each instance (deterministic, avoid similar colors)
            # Use a better color generation method
            def generate_distinct_color(inst_id, num_instances):
                """Generate distinct colors using HSV color space"""
                import colorsys
                hue = (inst_id * 137.508) % 360  # Golden angle for good distribution
                saturation = 0.7 + (inst_id % 3) * 0.1  # Vary saturation
                value = 0.8 + (inst_id % 2) * 0.15  # Vary brightness
                rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
                return tuple(int(c * 255) for c in rgb)
            
            instance_colors = {}
            for inst_id in unique_ids:
                if inst_id > 0:
                    color = generate_distinct_color(inst_id, num_instances)
                    instance_colors[inst_id] = color
                    instance_rgb[instance_map == inst_id] = color
                    print(f"   Instance {inst_id}: color RGB{color}")
            
            # Save RGB instance mask (format giống SegmentationObject)
            instance_output = args.output_path.replace(".png", "_instance.png")
            Image.fromarray(instance_rgb, mode='RGB').save(instance_output)
            print(f"Saved RGB instance mask to {instance_output} (format like SegmentationObject)")
            
            # Also save as uint16 for full ID range (nếu cần dùng ID gốc)
            instance_output_uint16 = args.output_path.replace(".png", "_instance_uint16.png")
            Image.fromarray(instance_map.astype(np.uint16), mode='I;16').save(instance_output_uint16)
            print(f"Saved instance ID map (uint16) to {instance_output_uint16}")
            
            # Save colored visualization (overlay trên semantic để dễ xem)
            instance_colored_output = args.output_path.replace(".png", "_instance_colored.png")
            Image.fromarray(instance_rgb, mode='RGB').save(instance_colored_output)
            print(f"Saved colored instance visualization to {instance_colored_output}")
    
    # Convert semantic mask to color
    color_mask = np.zeros((mask_pred.shape[0], mask_pred.shape[1], 3), dtype=np.uint8)
    for i, (name, color) in enumerate(zip(names, colors)):
        color_mask[mask_pred == i] = color
    
    # Save semantic segmentation
    Image.fromarray(color_mask).save(args.output_path)
    print(f"Saved semantic segmentation to {args.output_path}")
    print("Done!")

if __name__ == "__main__":
    main()

