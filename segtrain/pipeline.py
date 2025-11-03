import argparse, numpy as np, cv2
from pathlib import Path
from PIL import Image
import keras, tensorflow as tf

from segtrain.data.labelmap import read_labelmap
from segtrain.tta import TTAInference
from segtrain.postprocess import PostProcessor

def inference_pipeline(model_path: str, image_path: str, output_path: str,
                       labelmap_path: str, crop_size: int = 512,
                       use_tta: bool = True, use_postprocessing: bool = True):
    model = keras.models.load_model(model_path)
    names, colors = read_labelmap(Path(labelmap_path)); num_classes = len(names)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    img_np = np.asarray(img, dtype=np.float32)/255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    h,w = img_np.shape[:2]
    scale = min(crop_size / h, crop_size / w)
    new_h, new_w = int(h*scale), int(w*scale)
    img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = crop_size - new_h, crop_size - new_w
    img_padded = np.pad(img_resized, ((0,pad_h),(0,pad_w),(0,0)), mode="constant")

    if use_tta:
        tta = TTAInference(model)
        mask_pred, mask_probs = tta.predict(img_padded)
    else:
        outputs = model(tf.expand_dims(tf.constant(img_padded, dtype=tf.float32), 0), training=False)
        sem_logits = outputs[0] if isinstance(outputs, list) else (outputs.get("sem_logits") if isinstance(outputs, dict) else outputs)
        mask_probs = tf.nn.softmax(sem_logits, axis=-1).numpy()[0]
        mask_pred = np.argmax(mask_probs, axis=-1)

    mask_pred = mask_pred[:new_h, :new_w]
    mask_probs = mask_probs[:new_h, :new_w]

    mask_pred = cv2.resize(mask_pred.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST)
    mprobs_resized = np.zeros((H,W,num_classes), dtype=np.float32)
    for c in range(num_classes):
        mprobs_resized[:,:,c] = cv2.resize(mask_probs[:,:,c], (W,H), interpolation=cv2.INTER_LINEAR)

    if use_postprocessing:
        post = PostProcessor(use_morphology=True, min_blob_size=100, use_crf=False)
        img_original = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask_pred = post.process(img_original, mask_pred, mprobs_resized)

    color_mask = np.zeros((H,W,3), dtype=np.uint8)
    for i, (_, color) in enumerate(zip(names, colors)):
        color_mask[mask_pred == i] = color

    Image.fromarray(color_mask).save(output_path)
    print(f"Saved prediction to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--labelmap_path", required=True)
    ap.add_argument("--crop_size", type=int, default=512)
    ap.add_argument("--use_tta", action="store_true")
    ap.add_argument("--no_post", action="store_true")
    args = ap.parse_args()
    inference_pipeline(args.model_path, args.image_path, args.output_path, args.labelmap_path,
                       crop_size=args.crop_size, use_tta=args.use_tta, use_postprocessing=(not args.no_post))
