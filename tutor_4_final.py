import tensorflow as tf
import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math

'''Dataset: each sample is a color image of animal plus a pixel-wise mask (trimap)'''

'''
conda activate tensor_test
cd C:\Python\tutor-deep-learning\img_segment
code .  
'''

'''
Possible steps make model better:

'''

'''Explaination 
(B, H, W, C)
B — Batch size: how many images (or samples) are processed together.
H — Height: number of rows (pixels) in each image / feature map.
W — Width: number of columns (pixels).
C — Channels:
    For raw RGB images: C = 3 (R,G,B).
    For grayscale images: C = 1.
    For model feature maps: C = number of filters (learned channels).
    For segmentation outputs: C = number of classes (per-pixel logits or probs).
    For segmentation masks (indices): usually C = 1 (each pixel stores a class ID).

U-net model: semantic segmentation -> classify each pixel in a label

'''

'''
Possible way to improve Unet 
    mild geometric + photometric transform

    batch norm + ReLU, spatial dropout 2D

'''

#understand data: datapoint is a dictionary with img and mask of it

#load data 1 time, use forever
builder = tfds.builder('oxford_iiit_pet:4.0.0', data_dir=r"D:\dataset_deeplearning")
builder.download_and_prepare()
ds_train = builder.as_dataset(split='train[:80%]')
ds_val = builder.as_dataset(split='train[80%:]')
ds_test = builder.as_dataset(split='test')
info = builder.info

def resize(input_img, input_mask):
    # method: nearest-neighbor interpolation(pick the value of the closest pixel)
    # method: bilinear, set antialias=True when down-sampling for reduce jaggies
    input_img = tf.image.resize(input_img, (128, 128), method="bilinear", antialias=True)
    input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
    # Lock static shapes (helps tracing/perf)
    input_img.set_shape([128, 128, 3])
    input_mask.set_shape([128, 128, 1])
    return input_img, input_mask

#random flip img horizontally
def augment(input_img, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_img = tf.image.flip_left_right(input_img)
        input_mask = tf.image.flip_left_right(input_mask)

    return input_img, input_mask

def normalize(input_img, input_mask):
    input_img = tf.cast(input_img, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.int32)
    input_mask -= 1     #shift the mask class IDs down by 1
    return input_img, input_mask

#datapoint: 1 sample from my TFDS dataset
def load_img_train(datapoint):
   input_img = datapoint["image"]
   input_mask = datapoint["segmentation_mask"]
   input_img, input_mask = resize(input_img, input_mask)
   input_img, input_mask = augment(input_img, input_mask)
   input_img, input_mask = normalize(input_img, input_mask)

   return input_img, input_mask

def load_img_test(datapoint):
   input_img = datapoint["image"]
   input_mask = datapoint["segmentation_mask"]
   input_img, input_mask = resize(input_img, input_mask)
   input_img, input_mask = normalize(input_img, input_mask)

   return input_img, input_mask

AUTOTUNE = tf.data.AUTOTUNE
BATCH = 32
SHUFFLE = 1000

train_batch = (ds_train
                 .map(load_img_train, num_parallel_calls=AUTOTUNE)
                 .shuffle(SHUFFLE)                 # shuffle only train
                 .batch(BATCH)
                 .prefetch(AUTOTUNE))           # overlap CPU/GPU

val_batch  = (ds_val
                 .map(load_img_test, num_parallel_calls=AUTOTUNE)
                 .batch(BATCH)
                 .prefetch(AUTOTUNE))           # no shuffle for eval

test_batch  = (ds_test
                 .map(load_img_test, num_parallel_calls=AUTOTUNE)
                 .batch(BATCH)
                 .prefetch(AUTOTUNE))           # no shuffle for eval

def display(display_list):
    plt.figure(figsize=(12, 12))
    title = ["Input img", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()

# sample_batch = next(iter(train_batch))
# random_index = np.random.choice(sample_batch[0].shape[0])
# sample_img, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
# display([sample_img, sample_mask])

def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

#feature extraction with 2 conv -> return f, downsampling via max-pool + drop out -> return p
def downsample_block(x, n_filters):
    #If x has shape (B, H, W, C), then f is (B, H, W, n_filters).
    f = double_conv_block(x, n_filters)
    #p shape: (B, H/2, W/2, n_filters)
    p = layers.MaxPool2D(2) (f)
    p = layers.Dropout(0.3) (p)

    return f, p

'''If shape of x(B, H, W, C) and conv_feature is (B, 2H, 2W, F):
After Conv2DTranspose(..., stride=2): (B, 2H, 2W, n_filters)
After concatenate: (B, 2H, 2W, n_filters + F)
After double_conv_block: (B, 2H, 2W, n_filters)'''
def upsample_block(x, conv_feature, n_filters):
    #increase spatial size by 2x
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same") (x)
    x = layers.concatenate([x, conv_feature])
    x = layers.Dropout(0.3) (x)
    x = double_conv_block(x, n_filters)

    return x

#Those numbers (64, 128, 256, 512 — and 1024 at the bottleneck) are the 
# number of convolutional filters (channels) used at each stage of the U-Net
def build_unet_model():
    inputs = layers.Input(shape=(128, 128, 3))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    # unet model with Keras Functional API
    unet_model = keras.Model(inputs, outputs, name="U-Net")
    return unet_model

unet_model = build_unet_model()

unet_model.compile(optimizer=keras.optimizers.Adam(),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

NUM_EPOCHS = 10
BATCH = 32
TOTAL_TRAIN = info.splits["train"].num_examples
TRAIN_LEN   = int(0.8 * TOTAL_TRAIN)          # because you used train[:80%]
VAL_LEN     = TOTAL_TRAIN - TRAIN_LEN         # train[80%:]
# Or: get exact batch counts from the pipeline
# TRAIN_STEPS = tf.data.experimental.cardinality(train_batch).numpy()
# VAL_STEPS   = tf.data.experimental.cardinality(val_batch).numpy()

STEP_PER_EPOCH   = math.ceil(TRAIN_LEN / BATCH)
VALIDATION_STEPS = math.ceil(VAL_LEN   / BATCH)

history = unet_model.fit(
    train_batch,
    validation_data=val_batch,
    epochs=NUM_EPOCHS,
    steps_per_epoch=STEP_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
)

save_dir = r"C:\Python\tutor-deep-learning\img_segment\models"
unet_model.save(save_dir)

def create_mask(pred_mask):
    '''Takes the argmax over the channel (class) dimension.
        Converts per-pixel class scores (logits or softmax probabilities) into class IDs.
        Shape changes: (B, H, W, C) → (B, H, W).
        Each pixel now holds an integer in {0, 1, 2} for your 3 classes.'''
    pred_mask = tf.argmax(pred_mask, axis=-1)
    #Shape: (B, H, W) → (B, H, W, 1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset is not None:
        for image, mask in dataset.take(num):
            pred_mask = unet_model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        raise ValueError("Provide a dataset or define sample_img/sample_mask")
    
count = 0
for i in test_batch:
   count +=1
print("number of batches:", count)