# 🧪 Hướng Dẫn Test Model Trên Google Colab

## ✅ Có Cần Upload File Test Không?

**Câu trả lời:** **CÓ**, bạn cần upload file `test_model_colab.py` để test model trên Colab.

---

## 📤 Cách Upload và Test

### Bước 1: Upload File Test

**Cách A: Upload trực tiếp**
1. Mở Colab notebook
2. Click icon **folder** bên trái
3. Click **Upload**
4. Upload file `test_model_colab.py`

**Cách B: Copy code**
```python
# Cell 1: Tạo file test
%%writefile /content/test_model_colab.py
# Paste toàn bộ code từ test_model_colab.py vào đây
```

### Bước 2: Upload Model và Ảnh Test

**Model:**
- Model đã được lưu vào Drive khi train: `/content/drive/MyDrive/SegmentImg/models/attention_unet_focal_best.keras`
- Hoặc upload model mới vào Drive

**Ảnh test:**
- Upload ảnh test vào `/content/` hoặc Drive
- Hoặc dùng ảnh từ dataset: `/content/drive/MyDrive/SegmentImg/data/cheetah/JPEGImages/00000000_512resized.png`

### Bước 3: Chạy Test

**Cách 1: Chạy với defaults tự động (Khuyến nghị)**

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Import và chạy test
import sys
sys.path.insert(0, '/content')

from test_model_colab import main
main()  # Sẽ tự động tìm model và ảnh
```

**Cách 2: Chạy với arguments cụ thể**

```python
# Chạy với đường dẫn cụ thể
import sys
sys.argv = [
    'test_model_colab.py',
    '--model_path', '/content/drive/MyDrive/SegmentImg/models/attention_unet_focal_best.keras',
    '--image_path', '/content/drive/MyDrive/SegmentImg/data/cheetah/JPEGImages/00000000_512resized.png',
    '--output_dir', '/content/drive/MyDrive/SegmentImg/test_results',
    '--labelmap', '/content/labelmap.txt',
    '--save_boundary'
]

from test_model_colab import main
main()
```

**Cách 3: Chạy script trực tiếp**

```python
!python /content/test_model_colab.py \
    --model_path /content/drive/MyDrive/SegmentImg/models/attention_unet_focal_best.keras \
    --image_path /content/drive/MyDrive/SegmentImg/data/cheetah/JPEGImages/00000000_512resized.png \
    --output_dir /content/drive/MyDrive/SegmentImg/test_results \
    --labelmap /content/labelmap.txt \
    --save_boundary
```

---

## 📋 Quick Start

### Cell 1: Setup
```python
from google.colab import drive
drive.mount('/content/drive')

# Upload test_model_colab.py và labelmap.txt vào /content/ (dùng file browser)
```

### Cell 2: Test Model
```python
import sys
sys.path.insert(0, '/content')

from test_model_colab import main

# Test với defaults (tự động tìm model và ảnh)
main()
```

### Cell 3: Xem Kết Quả
```python
from IPython.display import Image, display
from pathlib import Path

output_dir = Path("/content/drive/MyDrive/SegmentImg/test_results")

# Hiển thị ảnh gốc
display(Image(str(output_dir / "pred_color.png")))
display(Image(str(output_dir / "pred_overlay.png")))
```

---

## 🎯 Các File Cần Upload

1. ✅ **test_model_colab.py** - Script test (bắt buộc)
2. ✅ **labelmap.txt** - File định nghĩa classes (bắt buộc)
3. ✅ **Model file (.keras)** - Model đã train (thường đã có trên Drive)
4. ✅ **Ảnh test** - Ảnh để test (optional, có thể dùng từ dataset)

---

## 📊 Kết Quả Test

Sau khi chạy, bạn sẽ có các file trong `output_dir`:

- `pred_index.png` - Mask dạng grayscale (0-255)
- `pred_color.png` - Mask đã colorize theo labelmap
- `pred_boundary.png` - Boundary heatmap (nếu có)
- `pred_overlay.png` - Overlay trên ảnh gốc

---

## 💡 Tips

1. **Tự động tìm model:** Script sẽ tự động tìm model ở các vị trí:
   - `/content/drive/MyDrive/SegmentImg/models/attention_unet_focal_best.keras`
   - `/content/drive/MyDrive/SegmentImg/models/unet_boundary_best.keras`
   - `/content/models/attention_unet_focal_best.keras`

2. **Tự động tìm ảnh:** Script sẽ tự động tìm ảnh từ dataset

3. **Test nhiều ảnh:** Chạy trong loop:
   ```python
   from pathlib import Path
   from test_model_colab import main
   import sys
   
   images = list(Path("/content/drive/MyDrive/SegmentImg/data/cheetah/JPEGImages").glob("*.png"))[:5]
   
   for img_path in images:
       sys.argv = [
           'test_model_colab.py',
           '--model_path', '/content/drive/MyDrive/SegmentImg/models/attention_unet_focal_best.keras',
           '--image_path', str(img_path),
           '--output_dir', f'/content/drive/MyDrive/SegmentImg/test_results/{img_path.stem}',
           '--labelmap', '/content/labelmap.txt'
       ]
       main()
   ```

---

## 🆘 Troubleshooting

### Lỗi: "Model không tồn tại"
- Kiểm tra model đã được lưu sau khi train
- Kiểm tra đường dẫn đúng
- Upload model vào Drive nếu cần

### Lỗi: "Ảnh không tồn tại"
- Upload ảnh test vào `/content/`
- Hoặc chỉ định đường dẫn đúng đến ảnh trong dataset

### Lỗi: "Labelmap không tồn tại"
- Upload `labelmap.txt` vào `/content/`
- Hoặc chỉ định đường dẫn đúng

---

**Chúc bạn test thành công! 🚀**

