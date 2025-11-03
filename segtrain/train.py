import os, keras, tensorflow as tf, numpy as np
from pathlib import Path

from segtrain.config import parse_args_with_defaults, check_paths
from segtrain.utils import set_seed
from segtrain.data.dataset import EnhancedMultiRootVOCDataset, make_tf_dataset, compute_class_weights
from segtrain.data.labelmap import read_labelmap
from segtrain.losses import sparse_ce_ignore_index, weighted_sparse_ce_ignore_index, focal_loss, tversky_loss
from segtrain.callbacks import EvalCallback

from segtrain.models.unet_boundary import build_unet_with_boundary
from segtrain.models.attention_unet import build_attention_unet
from segtrain.models.unetpp import build_unet_plusplus
from segtrain.models.backbone_unet import build_unet_with_backbone

def main_unet():
    args = parse_args_with_defaults()
    is_colab = (os.path.exists("/content") or "COLAB_GPU" in os.environ)

    roots = []
    for item in (args.data_roots or []):
        roots.extend([s for s in item.split(",") if s])
    if not roots:
        raise SystemExit("No valid data roots provided")

    check_paths(roots, args.labelmap, args.save_dir, is_colab=is_colab)
    set_seed(args.seed)

    names, colors = read_labelmap(Path(args.labelmap))
    num_classes = len(names)
    print(f"Classes ({num_classes}): {names}")

    train_ds_wrap = EnhancedMultiRootVOCDataset(
        roots=roots, image_set="train",
        names=names, colors=colors,
        crop_size=args.crop_size,
        use_advanced_aug=args.use_advanced_aug
    )
    val_ds_wrap = EnhancedMultiRootVOCDataset(
        roots=roots, image_set="val",
        names=names, colors=colors,
        crop_size=args.crop_size,
        use_advanced_aug=False
    )

    class_weights = None
    if args.loss == "weighted_ce":
        print("Computing class weights...")
        masks = []
        for i in range(min(100, len(train_ds_wrap))):
            _, m = train_ds_wrap.get_item(i)
            masks.append(m)
        class_weights = compute_class_weights(masks, num_classes, ignore_index=255)
        print(f"Class weights: {class_weights}")

    train_ds = make_tf_dataset(train_ds_wrap, batch_size=args.batch_size, shuffle=True, ignore_index=255)
    val_ds = make_tf_dataset(val_ds_wrap, batch_size=1, shuffle=False, ignore_index=255)

    if args.architecture == "unet":
        model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)
    elif args.architecture == "attention_unet":
        model = build_attention_unet(num_classes=num_classes, dropout=0.2)
    elif args.architecture == "unet_plusplus":
        model = build_unet_plusplus(num_classes=num_classes, dropout=0.2, deep_supervision=args.deep_supervision)
    elif args.architecture == "unet_backbone":
        model = build_unet_with_backbone(num_classes=num_classes, backbone=args.backbone,
                                         backbone_name=args.backbone_name, dropout=0.2)
    else:
        model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)

    if args.loss == "ce":
        sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)
    elif args.loss == "weighted_ce":
        sem_loss = weighted_sparse_ce_ignore_index(class_weights, ignore_index=255, from_logits=True)
    elif args.loss == "focal":
        sem_loss = focal_loss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=255, from_logits=True)
    elif args.loss == "tversky":
        sem_loss = tversky_loss(alpha=args.tversky_alpha, beta=args.tversky_beta, ignore_index=255, from_logits=True)
    else:
        sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)

    bce_logits = keras.losses.BinaryCrossentropy(from_logits=True)

    if args.architecture == "unet_plusplus" and args.deep_supervision:
        losses = {"ds1": sem_loss, "ds2": sem_loss, "ds3": sem_loss, "sem_logits": sem_loss, "boundary_logits": bce_logits}
        loss_weights = {"ds1": 0.25, "ds2": 0.25, "ds3": 0.25, "sem_logits": 1.0, "boundary_logits": 1.0}
    else:
        losses = {"sem_logits": sem_loss, "boundary_logits": bce_logits}
        loss_weights = {"sem_logits": 1.0, "boundary_logits": 1.0}

    optimizer = keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    ckpt_path = Path(args.save_dir) / f"{args.architecture}_{args.loss}_best.keras"
    eval_cb = EvalCallback(val_ds, num_classes=num_classes, ignore_index=255, ckpt_path=ckpt_path)

    model.fit(train_ds, epochs=args.epochs, callbacks=[eval_cb], verbose=1)
    print(f"\nTraining completed! Best model saved to: {ckpt_path}")

if __name__ == "__main__":
    main_unet()
