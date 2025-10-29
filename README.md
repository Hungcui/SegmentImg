## Dataset & Project Layout

**File data:** [Google Drive](https://drive.google.com/file/d/1sEXhk7ibO0AT6u-0w25figoL_YsvkWkv/view?usp=sharing)

### Recommended directory structure

```text
D:\
└─ animal_data\
   ├─ img_segment\
   │  ├─ model_train_v2.py          # training script
   │  └─ labelmap.txt               # class names + RGB colors
   ├─ models\
   │  └─ unet_boundary_best.keras   # best model (saved by EvalCallback)
   └─ data\
```
