# 📚 Hướng Dẫn Chi Tiết Train Model trên Kaggle GPU

## 🎯 Tổng Quan

Hướng dẫn này sẽ giúp bạn train model segmentation trên Kaggle với GPU (P100, T4, hoặc T4 x2) một cách chi tiết từng bước.

---

## 📋 Bước 1: Chuẩn Bị Dataset trên Kaggle

### ✅ Cách 1: Tạo Kaggle Dataset với File ZIP (Khuyến Nghị - Nhanh Nhất)

**Trên Kaggle:**

1. **Chuẩn bị file ZIP trên máy local:**
   - Zip toàn bộ folder `data/` (chứa các folder: cheetah, lion, wolf, tiger, hyena, fox)
   - File zip nên có cấu trúc: `data.zip` → `data/cheetah/`, `data/lion/`, ...
   - Hoặc zip toàn bộ, đảm bảo khi giải nén sẽ có folder `data/` ở root

2. **Truy cập:** https://www.kaggle.com/datasets
3. **Tạo dataset mới:** Click **"New Dataset"**
4. **Upload file ZIP:**
   - Kéo thả file `data.zip` (hoặc tên file zip của bạn)
   - Upload `labelmap.txt` vào cùng dataset
   - **Lưu ý:** Có thể upload nhiều file cùng lúc
5. **Đặt tên dataset:** Ví dụ: `animal-segmentation-dataset`
6. **Public hoặc Private:** Chọn theo nhu cầu
7. **Click "Create"**

**Ưu điểm:**
- ✅ Upload nhanh hơn (1 file thay vì nhiều folder)
- ✅ Giữ nguyên cấu trúc thư mục
- ✅ Dễ quản lý và chia sẻ

### Cách 2: Upload Từng Folder (Nếu không zip)

**Trên Kaggle:**

1. **Truy cập:** https://www.kaggle.com/datasets
2. **Tạo dataset mới:** Click **"New Dataset"**
3. **Upload data:**
   - Kéo thả hoặc chọn các folder chứa data:
     - `data/cheetah/`
     - `data/lion/`
     - `data/wolf/`
     - `data/tiger/`
     - `data/hyena/`
     - `data/fox/`
   - Upload `labelmap.txt` vào root của dataset
4. **Đặt tên dataset:** Ví dụ: `animal-segmentation-dataset`
5. **Public hoặc Private:** Chọn theo nhu cầu
6. **Click "Create"**

### Cách 3: Upload Trực Tiếp vào Notebook (Cho file nhỏ)

- Sử dụng file browser trong Kaggle notebook để upload trực tiếp
- File sẽ được lưu vào `/kaggle/working/`

---

## 🚀 Bước 2: Tạo Notebook trên Kaggle

1. **Truy cập:** https://www.kaggle.com/code
2. **Tạo notebook mới:** Click **"New Notebook"**
3. **Chọn GPU:**
   - Settings → Accelerator → **GPU** (P100, T4, hoặc 2xT4)
   - **Lưu ý:** Kaggle cho phép GPU miễn phí nhưng có giới hạn thời gian
4. **Add Dataset:**
   - Click **"Add Data"** bên phải
   - Tìm và chọn dataset bạn đã tạo ở Bước 1
   - Dataset sẽ được mount vào `/kaggle/input/YOUR_DATASET_NAME/`

---

## 🎯 So Sánh GPU: 2xT4 vs P100 - Nên Chọn Gì?

### 📊 Bảng So Sánh Chi Tiết

| Tiêu chí | **P100 (Single)** | **2xT4 (Dual)** | **Khuyến nghị** |
|----------|-------------------|-----------------|-----------------|
| **VRAM tổng** | 16GB | 32GB (16GB x2) | ⭐ 2xT4 cho model lớn |
| **VRAM per GPU** | 16GB | 16GB | = |
| **Compute Power** | Cao hơn T4 | Trung bình | ⭐ P100 cho tốc độ |
| **Batch Size lớn nhất** | 12-16 (512x512) | 20-32 (512x512) | ⭐ 2xT4 cho batch lớn |
| **Patch Size lớn nhất** | 512-640 | 768-1024 | ⭐ 2xT4 cho resolution cao |
| **Multi-GPU Setup** | ❌ Không cần | ✅ Cần config | P100 đơn giản hơn |
| **Training Speed** | ⚡ Nhanh hơn | 🐢 Chậm hơn (overhead) | ⭐ P100 nhanh hơn |
| **Độ phức tạp code** | ✅ Đơn giản | ⚠️ Cần multi-GPU strategy | ⭐ P100 dễ hơn |
| **Model lớn (EfficientNetB4+)** | ⚠️ Có thể OOM | ✅ Đủ VRAM | ⭐ 2xT4 cho model lớn |
| **Kaggle Availability** | Thường có | Ít hơn | P100 dễ kiếm hơn |

### 🎯 Khuyến Nghị Chọn GPU

#### ✅ **Chọn P100 khi:**
- ✅ Model nhỏ-trung bình (EfficientNetB0-B3)
- ✅ Muốn training nhanh
- ✅ Không muốn phức tạp code (single GPU)
- ✅ Patch size ≤ 640x640
- ✅ Batch size ≤ 16 là đủ
- ✅ **Đây là lựa chọn tốt nhất cho EfficientNetB3!**

#### ✅ **Chọn 2xT4 khi:**
- ✅ Model lớn (EfficientNetB4-B7, ResNet101+)
- ✅ Cần patch size lớn (768x768+)
- ✅ Cần batch size lớn (>20)
- ✅ Sẵn sàng config multi-GPU
- ✅ Dataset rất lớn, cần throughput cao

### 💡 Kết Luận cho EfficientNetB3

**Khuyến nghị: P100** ✅

**Lý do:**
1. ✅ EfficientNetB3 vừa phải, P100 đủ VRAM
2. ✅ Training nhanh hơn (không có overhead multi-GPU)
3. ✅ Code đơn giản hơn (single GPU)
4. ✅ Patch size 512x512 là tối ưu, P100 handle tốt
5. ✅ Batch size 8-12 đủ cho training ổn định

**Chỉ chọn 2xT4 nếu:**
- Bạn muốn train EfficientNetB4 trở lên
- Cần patch size ≥ 768x768
- Cần batch size ≥ 20

---

### 🚀 Multi-GPU Training với 2xT4 (Nếu Cần)

Nếu bạn chọn 2xT4, cần setup MirroredStrategy để sử dụng cả 2 GPU:

```python
# Setup Multi-GPU Strategy (chỉ cần nếu có 2xT4)
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    # Tạo MirroredStrategy để sử dụng tất cả GPU
    strategy = tf.distribute.MirroredStrategy()
    print(f"✅ Using {strategy.num_replicas_in_sync} GPU(s)")
    
    # Build model và train trong strategy scope
    with strategy.scope():
        # Build model ở đây
        model = build_unet_with_backbone(
            num_classes=num_classes,
            backbone="efficientnet",
            backbone_name="EfficientNetB3"
        )
        # Compile model
        model.compile(...)
        
    # Training sẽ tự động distribute qua các GPU
    model.fit(train_ds, epochs=EPOCHS, ...)
else:
    # Single GPU - không cần strategy
    model = build_unet_with_backbone(...)
    model.compile(...)
    model.fit(train_ds, epochs=EPOCHS, ...)
```

**Lưu ý Multi-GPU:**
- Batch size sẽ được chia đều cho các GPU (batch_size=16 → 8 per GPU)
- Effective batch size = batch_size × số_GPU
- Overhead communication có thể làm chậm 10-20%
- Chỉ nên dùng khi single GPU không đủ VRAM

---

## 🔧 Bước 3: Setup Môi Trường

### Cell 1: Kiểm tra GPU

```python
# Kiểm tra GPU có sẵn không
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpus)} GPU(s)")
print(f"GPU Details: {gpus}")

# Kiểm tra GPU details
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}: {gpu}")
        print(f"  Name: {gpu.name}")
        # Enable memory growth để tránh allocate toàn bộ VRAM
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✅ Memory growth enabled")
        except RuntimeError as e:
            print(f"  ⚠️ Cannot set memory growth: {e}")
    
    # Nếu có nhiều GPU, có thể dùng multi-GPU strategy
    if len(gpus) > 1:
        print(f"\n🚀 Multi-GPU detected: {len(gpus)} GPUs")
        print("💡 Tip: Để sử dụng multi-GPU, cần setup MirroredStrategy (xem phần Multi-GPU Training)")
    else:
        print(f"\n✅ Single GPU setup - Đơn giản và hiệu quả!")
else:
    print("⚠️ Không có GPU! Vui lòng chọn GPU trong Settings → Accelerator")
```

### Cell 2: Cài đặt Dependencies

```python
# Cài đặt các package cần thiết
!pip install -q tensorflow>=2.13.0
!pip install -q keras>=2.13.0
!pip install -q scikit-image
!pip install -q scipy
!pip install -q opencv-python
!pip install -q pillow

# Optional: CRF post-processing (nếu cần)
# !pip install -q git+https://github.com/lucasb-eyer/pydensecrf.git

print("✅ Đã cài đặt dependencies!")
```

### Cell 3: Giải Nén File ZIP Data (Nếu Dataset là File ZIP)

```python
import os
import zipfile
from pathlib import Path

# Tên dataset của bạn (thay YOUR_DATASET_NAME bằng tên thực tế)
DATASET_NAME = "YOUR_DATASET_NAME"  # Ví dụ: "animal-segmentation-dataset"
DATASET_PATH = f"/kaggle/input/{DATASET_NAME}"

# Tìm file zip trong dataset
zip_files = list(Path(DATASET_PATH).glob("*.zip"))
if not zip_files:
    # Thử các tên dataset khác
    possible_names = ["segmentimg", "animal-segmentation-dataset", "segmentation-data"]
    for name in possible_names:
        test_path = f"/kaggle/input/{name}"
        zip_files = list(Path(test_path).glob("*.zip"))
        if zip_files:
            DATASET_PATH = test_path
            DATASET_NAME = name
            break

if zip_files:
    zip_file = zip_files[0]  # Lấy file zip đầu tiên
    print(f"📦 Tìm thấy file ZIP: {zip_file}")
    
    # Thư mục giải nén (sẽ giải nén vào /kaggle/working/)
    EXTRACT_DIR = "/kaggle/working"
    
    # Kiểm tra xem đã giải nén chưa
    data_dir = Path(EXTRACT_DIR) / "data"
    if data_dir.exists() and any(data_dir.iterdir()):
        print("✅ Data đã được giải nén trước đó, bỏ qua...")
    else:
        print(f"📂 Đang giải nén {zip_file.name} vào {EXTRACT_DIR}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("✅ Giải nén hoàn tất!")
        
        # Kiểm tra cấu trúc sau khi giải nén
        if data_dir.exists():
            print(f"✅ Tìm thấy folder data tại: {data_dir}")
            subfolders = [d.name for d in data_dir.iterdir() if d.is_dir()]
            print(f"📁 Các folder trong data: {subfolders}")
        else:
            # Có thể file zip không có folder data ở root
            print("⚠️ Không tìm thấy folder 'data' sau khi giải nén")
            print("💡 Kiểm tra cấu trúc file zip của bạn")
else:
    print("⚠️ Không tìm thấy file ZIP trong dataset")
    print("💡 Nếu dataset đã là folder (không phải zip), bỏ qua bước này")
```

### Cell 4: Upload Code

**Option A: Upload file trực tiếp**
```python
# Sử dụng file browser bên trái để upload model_train_v3_kaggle.py
# File sẽ được lưu vào /kaggle/working/
```

**Option B: Copy code trực tiếp**
```python
# Tạo file mới
%%writefile /kaggle/working/model_train_v3_kaggle.py
# Paste toàn bộ code từ file model_train_v3_kaggle.py vào đây
```

**Option C: Nếu code đã ở trong dataset**
```python
# Copy từ input dataset
!cp /kaggle/input/YOUR_DATASET_NAME/model_train_v3_kaggle.py /kaggle/working/
!cp /kaggle/input/YOUR_DATASET_NAME/labelmap.txt /kaggle/working/
```

---

## 📁 Bước 5: Cấu Hình Đường Dẫn

### Cell 5: Thiết lập paths

```python
import os
from pathlib import Path

# Tìm dataset trong /kaggle/input
# Thay YOUR_DATASET_NAME bằng tên dataset của bạn
DATASET_NAME = "YOUR_DATASET_NAME"  # Ví dụ: "animal-segmentation-dataset"
DATASET_PATH = f"/kaggle/input/{DATASET_NAME}"

# Kiểm tra dataset có tồn tại không
if not Path(DATASET_PATH).exists():
    # Thử các tên khác
    possible_names = ["segmentimg", "animal-segmentation-dataset", "segmentation-data"]
    for name in possible_names:
        test_path = f"/kaggle/input/{name}"
        if Path(test_path).exists():
            DATASET_PATH = test_path
            DATASET_NAME = name
            break

# Xác định đường dẫn data (ưu tiên từ /kaggle/working/ nếu đã giải nén)
WORKING_DATA = Path("/kaggle/working/data")
INPUT_DATA = Path(f"{DATASET_PATH}/data")

# Kiểm tra data ở đâu (đã giải nén hay chưa)
if WORKING_DATA.exists() and any(WORKING_DATA.iterdir()):
    # Data đã được giải nén vào /kaggle/working/data
    DATA_BASE = "/kaggle/working"
    print("✅ Sử dụng data đã giải nén từ /kaggle/working/data")
elif INPUT_DATA.exists():
    # Data ở trong dataset (chưa zip hoặc đã giải nén khác)
    DATA_BASE = DATASET_PATH
    print(f"✅ Sử dụng data từ dataset: {DATASET_PATH}")
else:
    # Fallback: tìm trong các vị trí khác
    DATA_BASE = DATASET_PATH
    print("⚠️ Không tìm thấy folder data, kiểm tra lại cấu trúc dataset")

# Các dataset folders
DATA_ROOTS = [
    f"{DATA_BASE}/data/cheetah",
    f"{DATA_BASE}/data/lion",
    f"{DATA_BASE}/data/wolf",
    f"{DATA_BASE}/data/tiger",
    f"{DATA_BASE}/data/hyena",
    f"{DATA_BASE}/data/fox",
]

# Labelmap từ dataset hoặc working directory
LABELMAP_PATH = f"{DATASET_PATH}/labelmap.txt"
if not Path(LABELMAP_PATH).exists():
    LABELMAP_PATH = "/kaggle/working/labelmap.txt"

# Thư mục lưu model (luôn lưu vào /kaggle/working/)
SAVE_DIR = "/kaggle/working/models"

# Kiểm tra paths
print("\n" + "="*60)
print("CHECKING DATA PATHS")
print("="*60)
print(f"Dataset: {DATASET_NAME}")
print(f"Data base: {DATA_BASE}")

all_exist = True
for root in DATA_ROOTS:
    root_path = Path(root)
    if root_path.exists():
        jpeg_path = root_path / "JPEGImages"
        n_images = len(list(jpeg_path.glob("*"))) if jpeg_path.exists() else 0
        print(f"✅ {root_path.name}: {n_images} images")
    else:
        print(f"❌ {root_path.name} - KHÔNG TỒN TẠI!")
        all_exist = False

if Path(LABELMAP_PATH).exists():
    print(f"✅ Labelmap: {LABELMAP_PATH}")
else:
    print(f"❌ Labelmap không tồn tại: {LABELMAP_PATH}")
    print("💡 Hãy upload labelmap.txt vào dataset hoặc /kaggle/working/")
    all_exist = False

if not all_exist:
    print("\n⚠️ Một số paths không tồn tại!")
    print("💡 Kiểm tra lại:")
    print("   1. Đã giải nén file ZIP chưa? (Cell 3)")
    print("   2. Tên dataset có đúng không?")
    print("   3. Cấu trúc folder data có đúng không?")
else:
    print("\n✅ Tất cả paths đều hợp lệ!")
print("="*60)
```

---

## 🎓 Bước 6: Import Code và Train

### Cell 6: Import và Setup

```python
import sys

# CRITICAL: Set mixed precision policy to float32 BEFORE importing training code
# This prevents dtype conflicts when loading EfficientNet backbones
import tensorflow as tf
from keras import mixed_precision

try:
    mixed_precision.set_global_policy('float32')
    print("✅ Mixed precision policy set to float32")
except:
    tf.keras.backend.set_floatx('float32')
    print("✅ TensorFlow dtype set to float32")

# Disable mixed precision graph rewrite (for Kaggle/Colab environments)
try:
    tf.config.experimental.enable_mixed_precision_graph_rewrite(False)
    print("✅ Mixed precision graph rewrite disabled")
except:
    pass

# Thêm path để import code
sys.path.insert(0, '/kaggle/working')  # Code đã upload vào /kaggle/working

# Import code (phải import SAU KHI set mixed precision policy)
from model_train_v3_kaggle import (
    read_labelmap, EnhancedMultiRootVOCDataset, 
    make_tf_dataset, build_attention_unet, build_unet_with_boundary,
    build_unet_plusplus, build_unet_with_backbone,
    sparse_ce_ignore_index, weighted_sparse_ce_ignore_index,
    focal_loss, tversky_loss, compute_class_weights,
    EvalCallback
)
import keras
import numpy as np
import random
from pathlib import Path

print("✅ Đã import code!")
```

### Cell 7: Cấu hình Training

```python
# Cấu hình training
EPOCHS = 200  # Kaggle cho phép train lâu hơn, có thể tăng epochs
BATCH_SIZE = 8  # P100/T4: 8-16, tùy VRAM và crop_size
LR = 1e-3
CROP_SIZE = 512
ARCHITECTURE = "unet_backbone"  # 'unet', 'attention_unet', 'unet_plusplus', 'unet_backbone'
BACKBONE_NAME = "EfficientNetB3"  # 'EfficientNetB0', 'EfficientNetB3', 'EfficientNetB4'
LOSS = "focal"  # 'ce', 'weighted_ce', 'focal', 'tversky'
USE_ADVANCED_AUG = True
DEEP_SUPERVISION = False  # Chỉ dùng với unet_plusplus

# CRITICAL: Không enable mixed precision ở đây vì đã set float32 trong Cell 6
# Mixed precision sẽ gây conflict với EfficientNet backbone

# Seed
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print(f"Config:")
print(f"  Architecture: {ARCHITECTURE}")
if ARCHITECTURE == "unet_backbone":
    print(f"  Backbone: {BACKBONE_NAME}")
print(f"  Loss: {LOSS}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Crop size: {CROP_SIZE}")
```

### 📐 Khuyến Nghị Patch Size cho EfficientNetB3 trên GPU P100

**Cho EfficientNetB3 + U-Net decoder trên GPU P100 (16GB VRAM):**

| Patch Size | Batch Size | Ưu điểm | Nhược điểm | Khuyến nghị |
|------------|------------|---------|-------------|-------------|
| **384x384** | 16-20 | ✅ Batch lớn, train nhanh<br>✅ Tiết kiệm VRAM | ❌ Độ phân giải thấp<br>❌ Có thể mất chi tiết | Khi cần train nhanh |
| **512x512** | 8-12 | ✅ Cân bằng tốt<br>✅ Độ phân giải đủ<br>✅ Batch size hợp lý | - | **⭐ Khuyến nghị chính** |
| **640x640** | 4-6 | ✅ Độ phân giải cao<br>✅ Chi tiết tốt hơn | ❌ Batch nhỏ<br>❌ Train chậm hơn | Khi cần độ chính xác cao |
| **768x768** | 2-4 | ✅ Độ phân giải rất cao | ❌ Batch rất nhỏ<br>❌ Có thể OOM | Chỉ khi cần thiết |

**Cấu hình khuyến nghị cho EfficientNetB3:**
```python
# Cho EfficientNetB3 trên P100
CROP_SIZE = 512      # Patch size tối ưu
BATCH_SIZE = 8      # Batch size phù hợp với 512x512
ARCHITECTURE = "unet_backbone"
BACKBONE_NAME = "EfficientNetB3"
```

**Lưu ý:**
- EfficientNetB3 lớn hơn B0 (~12M params vs ~5M), cần nhiều VRAM hơn
- Nếu gặp OOM (Out of Memory), giảm batch_size hoặc giảm crop_size xuống 384
- Nếu VRAM còn dư, có thể tăng crop_size lên 640 để cải thiện chất lượng

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
# CRITICAL: Đảm bảo mixed precision policy vẫn là float32
# KHÔNG enable mixed_float16 ở đây vì sẽ gây conflict với EfficientNet
current_policy = str(mixed_precision.global_policy())
print(f"Current mixed precision policy: {current_policy}")
if 'float32' not in current_policy.lower():
    print("⚠️  Warning: Policy is not float32! Resetting to float32...")
    mixed_precision.set_global_policy('float32')

# Build model
if ARCHITECTURE == "unet":
    model = build_unet_with_boundary(num_classes=num_classes, dropout=0.2)
elif ARCHITECTURE == "attention_unet":
    model = build_attention_unet(num_classes=num_classes, dropout=0.2)
elif ARCHITECTURE == "unet_plusplus":
    model = build_unet_plusplus(num_classes=num_classes, dropout=0.2, deep_supervision=DEEP_SUPERVISION)
elif ARCHITECTURE == "unet_backbone":
    model = build_unet_with_backbone(
        num_classes=num_classes, 
        backbone="efficientnet", 
        backbone_name=BACKBONE_NAME,  # Sử dụng biến BACKBONE_NAME từ Cell 7
        dropout=0.2
    )
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
# Lưu ý: EvalCallback tự động thêm các metrics vào logs:
# - val_loss: negative mIoU (để ReduceLROnPlateau monitor)
# - val_miou: mean Intersection over Union
# - val_pa: Pixel Accuracy
# - val_bce: Binary Cross Entropy (cho boundary)

# Custom callback để lưu checkpoint mỗi 10 epochs
class PeriodicCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, period=10):
        super().__init__()
        self.filepath = filepath
        self.period = period
    
    def on_epoch_end(self, epoch, logs=None):
        # Lưu vào epoch 10, 20, 30, ... (epoch là 0-indexed, nên epoch+1)
        if (epoch + 1) % self.period == 0:
            filepath = self.filepath.format(epoch=epoch + 1)
            self.model.save(filepath)
            print(f"Saved checkpoint: {filepath}")

# Lưu checkpoint mỗi 10 epochs
periodic_checkpoint_cb = PeriodicCheckpoint(
    filepath=str(Path(SAVE_DIR) / f"{ARCHITECTURE}_{LOSS}_epoch{{epoch:02d}}.keras"),
    period=10
)

# Giảm learning rate khi không cải thiện
# EvalCallback sẽ tự động thêm val_loss vào logs (tính từ negative mIoU)
# Khi val_loss không giảm trong 5 epochs liên tiếp, LR sẽ giảm 50%
lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # EvalCallback tự động thêm metric này vào logs
    factor=0.5,          # Giảm LR còn 50% khi không cải thiện
    patience=5,          # Đợi 5 epochs không cải thiện
    min_lr=1e-6,         # LR tối thiểu
    verbose=1             # Hiển thị thông báo khi giảm LR
)

# TensorBoard (optional)
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=str(Path(SAVE_DIR) / "logs"),
    histogram_freq=1
)

print("Starting training...")
print(f"Save directory: {SAVE_DIR}")
print("Best model will be saved automatically by EvalCallback")
print("Checkpoints will be saved every 10 epochs")
print("Learning rate will be reduced automatically when val_loss plateaus")

# Train
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[eval_cb, periodic_checkpoint_cb, lr_callback, tensorboard_cb],
    verbose=1
)

print(f"\n✅ Training completed!")
print(f"Best model: {ckpt_path}")
```

---

## 💾 Bước 7: Lưu và Tải Model

### Lưu model

```python
# Model đã được lưu tự động bởi callbacks vào /kaggle/working/models/
# Best model: {SAVE_DIR}/{ARCHITECTURE}_{LOSS}_best.keras
# Checkpoints: {SAVE_DIR}/{ARCHITECTURE}_{LOSS}_epochXX.keras

# File trong /kaggle/working/ sẽ được lưu tự động khi commit notebook
```

### Tải model để inference

```python
# Load model đã train
model_path = f"{SAVE_DIR}/{ARCHITECTURE}_{LOSS}_best.keras"
model = keras.models.load_model(model_path)
print("✅ Model loaded!")
```

### Download Model về máy local

```python
# Trong Kaggle notebook, file trong /kaggle/working/ sẽ tự động được lưu khi commit
# Hoặc download thủ công:
from IPython.display import FileLink
FileLink(f"{SAVE_DIR}/{ARCHITECTURE}_{LOSS}_best.keras")
```

---

## 🧪 Bước 7.5: Test Model

### Upload File Test Model

**Cách 1: Upload qua File Browser**
- Click vào file browser bên trái
- Upload file `test_model_kaggle.py` vào `/kaggle/working/`

**Cách 2: Copy code trực tiếp**
```python
# Tạo file test_model_kaggle.py
%%writefile /kaggle/working/test_model_kaggle.py
# Paste toàn bộ code từ file test_model_kaggle.py vào đây
```

### Cell Test Model: Chạy Test với Defaults

```python
# Import và chạy test script
import sys
sys.path.insert(0, '/kaggle/working')

from test_model_kaggle import main

# Chạy với defaults (tự động tìm model, image, labelmap)
main()
```

### Cell Test Model: Chạy Test với Arguments

```python
# Import và chạy test script với arguments cụ thể
import sys
sys.path.insert(0, '/kaggle/working')

from test_model_kaggle import main
import sys

# Set arguments
sys.argv = [
    'test_model_kaggle.py',
    '--model_path', '/kaggle/working/models/attention_unet_focal_best.keras',
    '--image_path', '/kaggle/working/data/cheetah/JPEGImages/00000000.jpg',
    '--output_dir', '/kaggle/working/test_results',
    '--labelmap', '/kaggle/working/labelmap.txt',
    '--save_boundary'
]

main()
```

### Hoặc chạy trực tiếp từ command line

```python
# Chạy script như một chương trình Python
!python /kaggle/working/test_model_kaggle.py \
    --model_path /kaggle/working/models/attention_unet_focal_best.keras \
    --image_path /kaggle/working/data/cheetah/JPEGImages/00000000.jpg \
    --output_dir /kaggle/working/test_results \
    --labelmap /kaggle/working/labelmap.txt \
    --save_boundary
```

### Xem kết quả

```python
# Hiển thị các file kết quả
from pathlib import Path
from IPython.display import Image, display

output_dir = Path("/kaggle/working/test_results")

if output_dir.exists():
    print("📁 Files trong test_results:")
    for file in output_dir.glob("*.png"):
        print(f"   - {file.name}")
        
    # Hiển thị một số kết quả
    if (output_dir / "pred_color.png").exists():
        print("\n🖼️  Colorized prediction:")
        display(Image(str(output_dir / "pred_color.png")))
    
    if (output_dir / "pred_overlay.png").exists():
        print("\n🖼️  Overlay prediction:")
        display(Image(str(output_dir / "pred_overlay.png")))
else:
    print("❌ Thư mục test_results chưa tồn tại")
    print("💡 Hãy chạy test script trước")
```

### Download kết quả về máy local

```python
# Tạo link download cho các file kết quả
from IPython.display import FileLink

output_dir = Path("/kaggle/working/test_results")
if output_dir.exists():
    for file in output_dir.glob("*.png"):
        print(f"📥 Download {file.name}:")
        display(FileLink(str(file)))
```

---

## 📊 Bước 8: Monitor Training

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

1. **Kaggle Session Limits:**
   - Free tier: ~30 giờ GPU/tuần
   - Session timeout: ~9 giờ
   - Tự động lưu khi commit notebook

2. **Lưu Model:**
   - ⚠️ **LUÔN lưu model vào `/kaggle/working/`** (tự động lưu khi commit)
   - Model được lưu tự động bởi callbacks vào `SAVE_DIR`
   - File trong `/kaggle/working/` sẽ được lưu khi bạn commit notebook

3. **GPU Limits:**
   - Free tier: ~30 giờ GPU/tuần
   - Có thể mua Kaggle Pro để có nhiều GPU time hơn
   - GPU sẽ tự động disconnect sau ~9 giờ

4. **Batch Size:**
   - P100 GPU: batch_size = 16-32
   - T4 GPU: batch_size = 16-24
   - T4 x2 GPU: batch_size = 32-48
   - Điều chỉnh theo VRAM của GPU

5. **Data Size:**
   - Dataset có thể lên đến 20GB (free tier) hoặc 100GB (Pro)
   - Upload dataset một lần, dùng lại nhiều lần
   - File trong `/kaggle/input/` là read-only

6. **Internet Access:**
   - Kaggle notebook có internet access để download weights từ ImageNet
   - Không cần lo về việc download pre-trained models

---

## 🎯 Quick Start - Chạy Script Tự Động

Nếu muốn chạy nhanh, dùng script tự động:

```python
# Chạy script tự động (tự động detect Kaggle và set paths)
exec(open('/kaggle/working/model_train_v3_kaggle.py').read())

# Hoặc gọi hàm main
from model_train_v3_kaggle import main_unet
main_unet()
```

---

## 📞 Troubleshooting

### Lỗi: "No GPU available"
- Settings → Accelerator → GPU → Save
- Đảm bảo notebook đang ở chế độ GPU (không phải CPU)

### Lỗi: "Out of memory"
- Giảm `BATCH_SIZE` xuống 8-12
- Giảm `CROP_SIZE` xuống 256
- Sử dụng gradient checkpointing

### Lỗi: "Dataset not found"
- Kiểm tra đã add dataset vào notebook chưa
- Kiểm tra tên dataset trong code có đúng không
- Dataset path: `/kaggle/input/YOUR_DATASET_NAME/`

### Lỗi: "Data folder not found sau khi giải nén"
- Kiểm tra cấu trúc file ZIP:
  - File ZIP nên chứa folder `data/` ở root
  - Hoặc khi giải nén sẽ tạo folder `data/`
- Kiểm tra Cell 3 (giải nén) đã chạy thành công chưa
- Xem log giải nén để biết file được giải nén vào đâu

### Model không được lưu
- Kiểm tra `SAVE_DIR` có tồn tại không
- Đảm bảo đang lưu vào `/kaggle/working/`
- File sẽ được lưu khi commit notebook

### Kaggle session timeout
- Kaggle tự động lưu file trong `/kaggle/working/` khi commit
- Model sẽ được lưu tự động bởi callbacks
- Có thể resume training bằng cách load checkpoint

---

## ✅ Checklist Trước Khi Train

- [ ] Đã zip folder `data/` thành file ZIP
- [ ] Đã tạo dataset trên Kaggle và upload file ZIP + `labelmap.txt`
- [ ] Đã tạo notebook mới
- [ ] Đã chọn GPU trong Settings → Accelerator
- [ ] Đã add dataset vào notebook
- [ ] Đã upload code (`model_train_v3_kaggle.py`) và `labelmap.txt` (nếu chưa có trong dataset)
- [ ] Đã chạy Cell 3 để giải nén file ZIP (nếu dataset là file ZIP)
- [ ] Đã kiểm tra data paths trong Cell 5
- [ ] Đã cài đặt dependencies
- [ ] Đã cấu hình `SAVE_DIR` để lưu vào `/kaggle/working/`
- [ ] Đã kiểm tra batch size phù hợp với GPU

---

## 🔄 So Sánh Kaggle vs Colab

| Tính năng | Kaggle | Colab |
|-----------|--------|-------|
| GPU Time | ~30h/tuần (free) | ~12h/ngày (free) |
| Session Timeout | ~9 giờ | ~90 phút idle |
| Data Storage | Dataset (20GB free) | Google Drive (15GB free) |
| Auto Save | Tự động khi commit | Phải mount Drive |
| Internet Access | ✅ Có | ✅ Có |
| GPU Types | P100, T4, T4 x2 | T4, A100 |

---

**Chúc bạn train thành công trên Kaggle! 🚀**

