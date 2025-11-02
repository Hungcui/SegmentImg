# 📚 Hướng Dẫn Chi Tiết Train Model trên Google Colab GPU

## 🎯 Tổng Quan

Hướng dẫn này sẽ giúp bạn train model segmentation trên Google Colab với GPU (T4 hoặc A100) một cách chi tiết từng bước.

---

## 📋 Bước 1: Upload Code Lên Colab

### ✅ Cách 1: Upload Trực Tiếp (Đơn Giản Nhất - Không Cần Git)

**Trên Colab:**

1. **Mở Colab:** https://colab.research.google.com/
2. **Tạo notebook mới**
3. **Upload files:**
   - Click icon **folder** bên trái (Files)
   - Click icon **Upload** (hoặc kéo thả)
   - Upload các file:
     - `model_train_v3_improved.py`
     - `inference_improved.py`
     - `labelmap.txt`

**Hoặc copy code trực tiếp:**
```python
# Cell 1: Tạo file
%%writefile /content/model_train_v3_improved.py
# Paste toàn bộ code từ file model_train_v3_improved.py vào đây
```

### Cách 2: Clone từ GitHub (Nếu Muốn Dùng Git)

**Trên máy local:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Trên Colab:**
```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

---

## 🚀 Bước 3: Tạo Notebook trên Colab

1. **Truy cập:** https://colab.research.google.com/
2. **Tạo notebook mới:** File → New notebook
3. **Chọn GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator: **GPU** (T4 hoặc A100)
   - Save

---

## 🔧 Bước 4: Setup Môi Trường

### Cell 1: Kiểm tra GPU

```python
# Kiểm tra GPU có sẵn không
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Kiểm tra GPU details
if tf.config.list_physical_devices('GPU'):
    gpu = tf.config.list_physical_devices('GPU')[0]
    print(f"GPU Name: {gpu}")
    # Enable memory growth
    tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ Không có GPU! Vui lòng chọn GPU trong Runtime -> Change runtime type")
```

### Cell 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Sau khi mount, bạn sẽ thấy đường dẫn:
# /content/drive/MyDrive/SegmentImg/
```

### Cell 3: Cài đặt Dependencies

```python
# Cài đặt các package cần thiết
!pip install -q tensorflow>=2.13.0
!pip install -q keras>=2.13.0
!pip install -q scikit-image
!pip install -q scipy
!pip install -q opencv-python
!pip install -q pillow

# Optional: CRF post-processing (nếu cần)
# !pip install -q pydensecrf

print("✅ Đã cài đặt dependencies!")
```

### Cell 4: Upload Code (Nếu Chưa Upload)

**Nếu bạn chưa upload code ở bước 1:**

**Option A: Upload file trực tiếp**
```python
# Sử dụng file browser bên trái để upload model_train_v3_improved.py
# Hoặc dùng code sau:
from google.colab import files
uploaded = files.upload()  # Chọn file model_train_v3_improved.py và labelmap.txt
```

**Option B: Copy code trực tiếp**
```python
# Tạo file mới
%%writefile /content/model_train_v3_improved.py
# Paste toàn bộ code từ file model_train_v3_improved.py vào đây
```

**Option C: Nếu code đã ở trên Drive**
```python
# Copy từ Drive
!cp /content/drive/MyDrive/SegmentImg/model_train_v3_improved.py /content/
!cp /content/drive/MyDrive/SegmentImg/labelmap.txt /content/
```

---

## 📁 Bước 5: Cấu Hình Đường Dẫn

### Cell 5: Thiết lập paths

```python
import os
from pathlib import Path

# Điều chỉnh đường dẫn theo cấu trúc của bạn
# Data ở Google Drive:
DATA_ROOT = "/content/drive/MyDrive/SegmentImg/data"
# Labelmap từ file đã upload:
LABELMAP_PATH = "/content/labelmap.txt"
# Hoặc nếu labelmap ở Drive:
# LABELMAP_PATH = "/content/drive/MyDrive/SegmentImg/labelmap.txt"

# Các dataset folders
DATA_ROOTS = [
    f"{DATA_ROOT}/cheetah",
    f"{DATA_ROOT}/lion",
    f"{DATA_ROOT}/wolf",
    f"{DATA_ROOT}/tiger",
    f"{DATA_ROOT}/hyena",
    f"{DATA_ROOT}/fox",
]

# Thư mục lưu model (khuyến nghị lưu vào Drive)
SAVE_DIR = "/content/drive/MyDrive/SegmentImg/models"

# Kiểm tra paths
print("Checking data paths...")
for root in DATA_ROOTS:
    if Path(root).exists():
        print(f"✅ {root}")
    else:
        print(f"❌ {root} - KHÔNG TỒN TẠI!")

if Path(LABELMAP_PATH).exists():
    print(f"✅ Labelmap: {LABELMAP_PATH}")
else:
    print(f"❌ Labelmap không tồn tại: {LABELMAP_PATH}")
```

---

## 🎓 Bước 6: Import Code và Train

### Cell 6: Import và Setup

```python
import sys

# Thêm path để import code
sys.path.insert(0, '/content')  # Code đã upload vào /content

# Import code
from model_train_v3_improved import (
    read_labelmap, EnhancedMultiRootVOCDataset, 
    make_tf_dataset, build_attention_unet, build_unet_with_boundary,
    build_unet_plusplus, build_unet_with_backbone,
    sparse_ce_ignore_index, weighted_sparse_ce_ignore_index,
    focal_loss, tversky_loss, compute_class_weights,
    EvalCallback
)
import tensorflow as tf
import keras
import numpy as np
import random
from pathlib import Path

print("✅ Đã import code!")
```

### Cell 7: Cấu hình Training

```python
# Cấu hình training
EPOCHS = 50  # Tăng số epochs khi train trên GPU
BATCH_SIZE = 8  # T4: 8-16, A100: 16-32
LR = 1e-3
CROP_SIZE = 512
ARCHITECTURE = "attention_unet"  # 'unet', 'attention_unet', 'unet_plusplus', 'unet_backbone'
LOSS = "focal"  # 'ce', 'weighted_ce', 'focal', 'tversky'
USE_ADVANCED_AUG = True
DEEP_SUPERVISION = False  # Chỉ dùng với unet_plusplus

# Seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print(f"Config:")
print(f"  Architecture: {ARCHITECTURE}")
print(f"  Loss: {LOSS}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
```

### Cell 8: Load Data và Build Model

```python
# Load labelmap
names, colors = read_labelmap(Path(LABELMAP_PATH))
num_classes = len(names)
print(f"Classes ({num_classes}): {names}")

# Build datasets
train_ds_wrap = EnhancedMultiRootVOCDataset(
    roots=DATA_ROOTS, image_set="train",
    names=names, colors=colors,
    crop_size=CROP_SIZE,
    use_advanced_aug=USE_ADVANCED_AUG
)
val_ds_wrap = EnhancedMultiRootVOCDataset(
    roots=DATA_ROOTS, image_set="val",
    names=names, colors=colors,
    crop_size=CROP_SIZE,
    use_advanced_aug=False
)

print(f"Train samples: {len(train_ds_wrap)}")
print(f"Val samples: {len(val_ds_wrap)}")

# Compute class weights if needed
class_weights = None
if LOSS == "weighted_ce":
    print("Computing class weights...")
    masks = []
    sample_size = min(100, len(train_ds_wrap))
    for i in range(sample_size):
        _, mask = train_ds_wrap.get_item(i)
        masks.append(mask)
    class_weights = compute_class_weights(masks, num_classes, ignore_index=255)
    print(f"Class weights: {class_weights}")

# Create tf.data datasets
train_ds = make_tf_dataset(train_ds_wrap, batch_size=BATCH_SIZE, shuffle=True, ignore_index=255)
val_ds = make_tf_dataset(val_ds_wrap, batch_size=1, shuffle=False, ignore_index=255)
```

### Cell 9: Build Model

```python
# Build model
if ARCHITECTURE == "unet":
    model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)
elif ARCHITECTURE == "attention_unet":
    model = build_attention_unet(num_classes=num_classes, dropout=0.2)
elif ARCHITECTURE == "unet_plusplus":
    model = build_unet_plusplus(num_classes=num_classes, dropout=0.2, deep_supervision=DEEP_SUPERVISION)
elif ARCHITECTURE == "unet_backbone":
    model = build_unet_with_backbone(num_classes=num_classes, backbone="efficientnet", 
                                    backbone_name="EfficientNetB0", dropout=0.2)
else:
    model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)

print(f"Model parameters: {model.count_params():,}")
model.summary()
```

### Cell 10: Setup Loss và Optimizer

```python
# Setup losses
if LOSS == "ce":
    sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)
elif LOSS == "weighted_ce":
    sem_loss = weighted_sparse_ce_ignore_index(class_weights, ignore_index=255, from_logits=True)
elif LOSS == "focal":
    sem_loss = focal_loss(alpha=0.25, gamma=2.0, ignore_index=255, from_logits=True)
elif LOSS == "tversky":
    sem_loss = tversky_loss(alpha=0.5, beta=0.5, ignore_index=255, from_logits=True)
else:
    sem_loss = sparse_ce_ignore_index(ignore_index=255, from_logits=True)

bce_logits = keras.losses.BinaryCrossentropy(from_logits=True)

# Handle multiple outputs for deep supervision
if ARCHITECTURE == "unet_plusplus" and DEEP_SUPERVISION:
    losses = {
        "ds1": sem_loss,
        "ds2": sem_loss,
        "ds3": sem_loss,
        "sem_logits": sem_loss,
        "boundary_logits": bce_logits
    }
    loss_weights = {
        "ds1": 0.25,
        "ds2": 0.25,
        "ds3": 0.25,
        "sem_logits": 1.0,
        "boundary_logits": 1.0
    }
else:
    losses = {
        "sem_logits": sem_loss,
        "boundary_logits": bce_logits
    }
    loss_weights = {"sem_logits": 1.0, "boundary_logits": 1.0}

optimizer = keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

print("✅ Model compiled!")
```

### Cell 11: Setup Callbacks và Train

```python
# Tạo thư mục lưu model
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Callbacks
ckpt_path = Path(SAVE_DIR) / f"{ARCHITECTURE}_{LOSS}_best.keras"
eval_cb = EvalCallback(val_ds, num_classes=num_classes, ignore_index=255, ckpt_path=ckpt_path)

# Lưu checkpoint định kỳ
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=str(Path(SAVE_DIR) / f"{ARCHITECTURE}_{LOSS}_epoch{{epoch:02d}}.keras"),
    save_freq='epoch',
    period=10,  # Lưu mỗi 10 epochs
    verbose=1
)

# Giảm learning rate khi không cải thiện
lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# TensorBoard (optional)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=str(Path(SAVE_DIR) / "logs"),
    histogram_freq=1
)

print("Starting training...")
print(f"Save directory: {SAVE_DIR}")

# Train
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[eval_cb, checkpoint_cb, lr_callback, tensorboard_cb],
    verbose=1
)

print(f"\n✅ Training completed!")
print(f"Best model: {ckpt_path}")
```

---

## 💾 Bước 6: Lưu và Tải Model

### Lưu model
```python
# Model đã được lưu tự động bởi callbacks
# Best model: {SAVE_DIR}/{ARCHITECTURE}_{LOSS}_best.keras
# Checkpoints: {SAVE_DIR}/{ARCHITECTURE}_{LOSS}_epochXX.keras
```

### Tải model để inference
```python
# Load model đã train
model_path = f"{SAVE_DIR}/{ARCHITECTURE}_{LOSS}_best.keras"
model = keras.models.load_model(model_path)
print("✅ Model loaded!")
```

---

## 📊 Bước 7: Monitor Training

### Xem TensorBoard
```python
# Trong Colab, chạy:
%load_ext tensorboard
%tensorboard --logdir {SAVE_DIR}/logs
```

### Xem training history
```python
import matplotlib.pyplot as plt

# Plot loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history.get('val_loss', []), label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history.get('sem_logits_loss', []), label='semantic')
plt.plot(history.history.get('boundary_logits_loss', []), label='boundary')
plt.title('Component Losses')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## ⚠️ Lưu Ý Quan Trọng

1. **Colab Disconnect:**
   - Colab sẽ disconnect sau ~90 phút không hoạt động
   - Giữ browser tab mở và thỉnh thoảng scroll để tránh disconnect
   - Hoặc sử dụng extension như "Colab Alive" để tự động refresh

2. **Lưu Model:**
   - ⚠️ **LUÔN lưu model vào Google Drive**, không lưu vào `/content` (sẽ mất khi disconnect)
   - Model được lưu tự động bởi callbacks vào `SAVE_DIR`

3. **GPU Limits:**
   - Free tier: ~12 giờ GPU/ngày
   - Nếu hết quota, đợi đến ngày hôm sau hoặc upgrade Colab Pro

4. **Batch Size:**
   - T4 GPU: batch_size = 8-16
   - A100 GPU: batch_size = 16-32
   - Điều chỉnh theo VRAM của GPU

5. **Data Size:**
   - Nếu dataset lớn, upload lên Drive và mount
   - Tránh upload trực tiếp vào Colab (có thể bị giới hạn)

---

## 🎯 Quick Start Commands

Nếu muốn chạy nhanh, copy toàn bộ code từ file `colab_train.py` vào một cell và chạy:

```python
# Chạy script tự động
exec(open('/content/colab_train.py').read())
```

Hoặc sử dụng script đã tạo:

```python
# Upload colab_train.py trước, sau đó:
import colab_train
```

---

## 📞 Troubleshooting

### Lỗi: "No GPU available"
- Runtime → Change runtime type → GPU → Save

### Lỗi: "Out of memory"
- Giảm `BATCH_SIZE`
- Giảm `CROP_SIZE`
- Sử dụng gradient checkpointing

### Lỗi: "Drive mount failed"
- Chạy lại cell mount Drive
- Đảm bảo cho phép Colab truy cập Drive

### Model không được lưu
- Kiểm tra `SAVE_DIR` có tồn tại không
- Đảm bảo đã mount Drive nếu lưu vào Drive

---

## ✅ Checklist Trước Khi Train

- [ ] Đã chọn GPU trong Runtime settings
- [ ] Đã mount Google Drive (nếu dùng)
- [ ] Đã upload code và data
- [ ] Đã kiểm tra data paths
- [ ] Đã cài đặt dependencies
- [ ] Đã cấu hình `SAVE_DIR` để lưu vào Drive
- [ ] Đã kiểm tra batch size phù hợp với GPU

---

**Chúc bạn train thành công! 🚀**

