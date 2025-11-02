# 📤 Hướng Dẫn Upload Data Lên Google Drive

## 🎯 Tổng Quan

Data cần được upload lên **Google Drive** để dùng trên Colab. Folder `data/` không được commit vào Git vì quá lớn.

---

## 📋 Cấu Trúc Data Cần Upload

```
data/
├── cheetah/
│   ├── JPEGImages/
│   │   ├── 00000000_512resized.png
│   │   └── ...
│   ├── SegmentationClass/
│   │   ├── 00000000.png
│   │   └── ...
│   └── ImageSets/
│       └── Segmentation/
│           ├── train.txt
│           └── val.txt
├── lion/
│   └── ... (cùng cấu trúc)
├── wolf/
├── tiger/
├── hyena/
└── fox/
```

---

## ✅ Cách 1: Upload Qua Trình Duyệt (Khuyến Nghị)

### Bước 1: Chuẩn Bị Data

1. Đảm bảo folder `data/` có đầy đủ các dataset:
   - cheetah, lion, wolf, tiger, hyena, fox
2. Mỗi dataset có cấu trúc:
   - `JPEGImages/` - Ảnh gốc
   - `SegmentationClass/` - Mask segmentation
   - `ImageSets/Segmentation/` - train.txt và val.txt

### Bước 2: Upload Lên Google Drive

1. **Truy cập Google Drive:**
   - Vào https://drive.google.com
   - Đăng nhập tài khoản Google

2. **Tạo folder mới:**
   - Click **"New"** → **"Folder"**
   - Đặt tên: `SegmentImg`
   - Click **"Create"**

3. **Upload folder data:**
   
   **Cách A: Upload từng dataset (Khuyến nghị cho dataset lớn)**
   - Vào folder `SegmentImg` vừa tạo
   - Click **"New"** → **"Folder"** → Đặt tên `data`
   - Vào folder `data`
   - Upload từng dataset một:
     - Kéo thả folder `cheetah` vào
     - Đợi upload xong
     - Tiếp tục với `lion`, `wolf`, `tiger`, `hyena`, `fox`
   
   **Cách B: Upload cả folder data (Nếu nhỏ < 10GB)**
   - Nén folder `data` thành file `.zip` hoặc `.rar`
   - Upload file `.zip` lên Drive
   - Giải nén trên Drive: Right-click → **"Open with"** → **"Google Drive"**

4. **Cấu trúc trên Drive:**
   ```
   MyDrive/
   └── SegmentImg/
       └── data/
           ├── cheetah/
           ├── lion/
           ├── wolf/
           ├── tiger/
           ├── hyena/
           └── fox/
   ```

### Bước 3: Kiểm Tra Upload

1. Vào folder `SegmentImg/data/` trên Drive
2. Kiểm tra có đủ 6 folders: cheetah, lion, wolf, tiger, hyena, fox
3. Vào một folder (ví dụ `cheetah`) kiểm tra:
   - Có folder `JPEGImages` với các file ảnh
   - Có folder `SegmentationClass` với các file mask
   - Có folder `ImageSets/Segmentation/` với `train.txt` và `val.txt`

---

## ✅ Cách 2: Upload Bằng Google Drive Desktop (Cho Dataset Lớn)

Nếu dataset quá lớn (>10GB), dùng Google Drive Desktop để sync:

### Bước 1: Cài Google Drive Desktop

1. Tải: https://www.google.com/drive/download/
2. Cài đặt và đăng nhập
3. Chọn folder muốn sync với Drive

### Bước 2: Sync Data

1. Copy folder `data/` vào folder Drive Desktop (thường ở `C:\Users\YourName\Google Drive`)
2. Google Drive Desktop sẽ tự động upload lên cloud
3. Đợi sync hoàn tất (có thể mất vài giờ tùy kích thước)

---

## ✅ Cách 3: Upload Từ Colab (Cho File Nhỏ)

Nếu chỉ cần upload vài file nhỏ:

```python
# Trên Colab
from google.colab import files
uploaded = files.upload()  # Chọn files cần upload

# Files sẽ được upload vào /content/
```

**Lưu ý:** Cách này chỉ dùng cho file nhỏ, không phù hợp cho dataset lớn.

---

## 🔗 Sử Dụng Data Trên Colab

### Bước 1: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Bước 2: Cấu Hình Paths

```python
from pathlib import Path

# Đường dẫn data trên Drive
DATA_ROOT = "/content/drive/MyDrive/SegmentImg/data"

# Kiểm tra data có tồn tại không
cheetah_path = Path(f"{DATA_ROOT}/cheetah")
if cheetah_path.exists():
    print("✅ Data đã được upload!")
    print(f"Cheetah images: {len(list((cheetah_path / 'JPEGImages').glob('*')))} files")
else:
    print("❌ Data chưa được upload hoặc đường dẫn sai!")
```

### Bước 3: Kiểm Tra Từng Dataset

```python
DATA_ROOTS = [
    f"{DATA_ROOT}/cheetah",
    f"{DATA_ROOT}/lion",
    f"{DATA_ROOT}/wolf",
    f"{DATA_ROOT}/tiger",
    f"{DATA_ROOT}/hyena",
    f"{DATA_ROOT}/fox",
]

print("Kiểm tra datasets:")
for root in DATA_ROOTS:
    path = Path(root)
    if path.exists():
        jpeg = path / "JPEGImages"
        seg = path / "SegmentationClass"
        imgset = path / "ImageSets" / "Segmentation"
        
        if jpeg.exists() and seg.exists() and imgset.exists():
            n_images = len(list(jpeg.glob("*")))
            n_masks = len(list(seg.glob("*")))
            print(f"✅ {path.name}: {n_images} images, {n_masks} masks")
        else:
            print(f"⚠️ {path.name}: Thiếu cấu trúc VOC")
    else:
        print(f"❌ {path.name}: Không tồn tại")
```

---

## ⚠️ Lưu Ý Quan Trọng

1. **Kích Thước Data:**
   - Google Drive miễn phí: 15GB
   - Nếu dataset > 15GB, cần mua thêm dung lượng hoặc chia nhỏ

2. **Thời Gian Upload:**
   - Dataset nhỏ (<1GB): vài phút
   - Dataset trung bình (1-5GB): 10-30 phút
   - Dataset lớn (>5GB): vài giờ

3. **Kiểm Tra Sau Upload:**
   - Luôn kiểm tra số lượng files
   - Đảm bảo không bị thiếu file
   - Kiểm tra cấu trúc folder đúng

4. **Lưu Model:**
   - Model cũng nên lưu vào Drive để không mất khi disconnect Colab
   - Tạo folder `models/` trong `SegmentImg/`

---

## 📊 Checklist Upload Data

- [ ] Đã tạo folder `SegmentImg` trên Drive
- [ ] Đã tạo folder `data` trong `SegmentImg`
- [ ] Đã upload đủ 6 datasets (cheetah, lion, wolf, tiger, hyena, fox)
- [ ] Mỗi dataset có đủ:
  - [ ] Folder `JPEGImages` với ảnh
  - [ ] Folder `SegmentationClass` với masks
  - [ ] Folder `ImageSets/Segmentation/` với train.txt và val.txt
- [ ] Đã kiểm tra số lượng files đúng
- [ ] Đã mount Drive trên Colab
- [ ] Đã kiểm tra paths trên Colab

---

## 🆘 Troubleshooting

### Lỗi: "Quota exceeded" khi upload

**Giải pháp:**
- Xóa files không cần thiết trên Drive
- Hoặc mua thêm dung lượng Google One

### Lỗi: Upload bị gián đoạn

**Giải pháp:**
- Upload từng dataset một
- Sử dụng Google Drive Desktop để resume
- Kiểm tra kết nối internet ổn định

### Lỗi: "Path not found" trên Colab

**Giải pháp:**
```python
# Kiểm tra đường dẫn
from pathlib import Path
print(Path("/content/drive/MyDrive/SegmentImg/data").exists())

# Liệt kê các folder
!ls -la "/content/drive/MyDrive/SegmentImg/"

# Tìm folder data
!find "/content/drive/MyDrive" -name "data" -type d
```

---

## ✅ Sau Khi Upload Xong

Trên Colab, chạy:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Kiểm tra data
DATA_ROOT = "/content/drive/MyDrive/SegmentImg/data"
from pathlib import Path

if Path(DATA_ROOT).exists():
    print("✅ Data đã sẵn sàng!")
    # Tiếp tục train model
else:
    print("❌ Chưa tìm thấy data, vui lòng kiểm tra lại!")
```

---

**Chúc bạn upload thành công! 🚀**

