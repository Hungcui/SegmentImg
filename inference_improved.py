# Improved Inference Script with TTA and Post-processing
# Usage: python inference_improved.py --model_path <path> --image_path <path> --output_path <path> --labelmap <path>

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import keras
from model_train_v3_improved import (
    read_labelmap, TTAInference, PostProcessor, inference_pipeline
)

def main():
    parser = argparse.ArgumentParser(description="Improved inference with TTA and post-processing")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--labelmap", type=str, required=True, help="Path to labelmap file")
    parser.add_argument("--crop_size", type=int, default=512, help="Input size for model")
    parser.add_argument("--use_tta", action="store_true", default=True, help="Use Test Time Augmentation")
    parser.add_argument("--use_postprocessing", action="store_true", default=True, help="Use post-processing")
    parser.add_argument("--min_blob_size", type=int, default=100, help="Minimum blob size to keep")
    parser.add_argument("--use_crf", action="store_true", help="Use CRF refinement")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    print(f"Processing image: {args.image_path}")
    print(f"TTA: {args.use_tta}, Post-processing: {args.use_postprocessing}")
    
    inference_pipeline(
        model_path=args.model_path,
        image_path=args.image_path,
        output_path=args.output_path,
        labelmap_path=args.labelmap,
        crop_size=args.crop_size,
        use_tta=args.use_tta,
        use_postprocessing=args.use_postprocessing
    )
    
    print(f"Done! Output saved to {args.output_path}")

if __name__ == "__main__":
    main()


