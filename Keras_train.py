# import tensorflow as tf
# import numpy as np
# import keras


import os



# 训练集


train_input_dir = "static/DataSet/train/images/"
train_target_dir = "static/DataSet/train/annotations/trimaps/"
# 验证集
val_input_dir = "static/DataSet/val/images/"
val_target_dir = "static/DataSet/val/annotations/trimaps/"
img_size = (160, 160)
num_classes = 3
batch_size = 32

# 训练
train_input_img_paths = sorted(
    [
        os.path.join(train_input_dir, fname)
        for fname in os.listdir(train_input_dir)
        if fname.endswith(".jpg")
    ]
)
train_target_img_paths = sorted(
    [
        os.path.join(train_target_dir, fname)
        for fname in os.listdir(train_target_dir)
        if fname.endswith(".png")
    ]
)
# 验证
val_input_img_paths = sorted(
    [
        os.path.join(val_input_dir, fname)
        for fname in os.listdir(val_input_dir)
        if fname.endswith(".jpg")
    ]
)
val_target_img_paths = sorted(
    [
        os.path.join(val_target_dir, fname)
        for fname in os.listdir(val_target_dir)
        if fname.endswith(".png")
    ]
)




from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import TensorBoard

class OxfordPets(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, [y,y]




from tensorflow.keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    # 160,160,3
    conv0_0 = layers.Conv2D(32,3,strides=1,padding="same",activation=keras.activations.relu)(inputs)
    conv0_0 = layers.Conv2D(32,3,strides=1,padding="same",activation=keras.activations.relu)(conv0_0)
    # 160,160,3  ->  160,160,32
    temporary0_0 = conv0_0
    pool0_0 = layers.MaxPooling2D(2,strides=2,padding="same")(conv0_0)
    # 160,160,32 ->  80,80,32
    conv1_0 = layers.Conv2D(64,3, strides=1, padding="same",activation=keras.activations.relu)(pool0_0)
    conv1_0 = layers.Conv2D(64,3, strides=1, padding="same",activation=keras.activations.relu)(conv1_0)
    # 80,80,32   ->  80,80,64
    temporary1_0 = conv1_0
    pool1_0 = layers.MaxPooling2D(2,strides=2,padding="same")(conv1_0)
    # 80,80,64   ->  40,40,64
    conv2_0 = layers.Conv2D(128,3, strides=1, padding="same",activation=keras.activations.relu)(pool1_0)
    conv2_0 = layers.Conv2D(128,3, strides=1, padding="same",activation=keras.activations.relu)(conv2_0)
    # 40,40,64   ->  40,40,128
    temporary2_0 = conv2_0
    pool2_0 = layers.MaxPooling2D(2,strides=2,padding="same")(conv2_0)
    # 40,40,128  ->  20,20,128
    conv3_0 = layers.Conv2D(256,3, strides=1, padding="same",activation=keras.activations.relu)(pool2_0)
    conv3_0 = layers.Conv2D(256,3, strides=1, padding="same",activation=keras.activations.relu)(conv3_0)
    #20,20,128   ->  20,20,256
    temporary3_0 = conv3_0
    pool3_0 = layers.MaxPooling2D(2,strides=2,padding="same")(conv3_0)
    #20,20,256   ->  10,10,256
    conv4_0 = layers.Conv2D(512,3, strides=1, padding="same",activation=keras.activations.relu)(pool3_0)
    conv4_0 = layers.Conv2D(512,3, strides=1, padding="same",activation=keras.activations.relu)(conv4_0)
    #10,10,256   ->  10,10,512

    conv3_1_up = layers.Conv2DTranspose(256,3,strides=1,padding="same",activation=keras.activations.relu)(conv4_0)
    #10,10,512   ->  10,10,256
    conv3_1_up = layers.UpSampling2D(2)(conv3_1_up)
    #10,10,256   ->  20,20,256
    conv3_1 = layers.Concatenate(axis=3)([temporary3_0,conv3_1_up])
    #20,20,256+20,20,256 -> 20,20,512
    conv3_1 = layers.Conv2D(256,3,strides=1,padding="same",activation=keras.activations.relu)(conv3_1)
    #20,20,512   ->  20,20,256

    conv2_1_up = layers.Conv2DTranspose(128,3,strides=1,padding="same",activation=keras.activations.relu)(temporary3_0)
    #20,20,256   ->  20,20,128
    conv2_1_up = layers.UpSampling2D(2)(conv2_1_up)
    #20,20,128   ->  40,40,128
    conv2_1 = layers.Concatenate(axis=3)([conv2_1_up,temporary2_0])
    #40,40,128+40,40,128 -> 40,40,256
    conv2_1 = layers.Conv2D(128,3,strides=1,padding="same",activation=keras.activations.relu)(conv2_1)
    #40,40,256  -> 40,40,128

    conv2_2_up = layers.Conv2DTranspose(128,3,strides=1,padding="same",activation=keras.activations.relu)(conv3_1)
    #20,20,256  -> 20,20,128
    conv2_2_up = layers.UpSampling2D(2)(conv2_2_up)
    #20,20,128  -> 40,40,128
    conv2_2 = layers.Concatenate(axis=3)([temporary2_0,conv2_1,conv2_2_up])
    #40,40,128 * 3  => 40,40,384
    conv2_2 = layers.Conv2D(128,3,strides=1,padding="same",activation=keras.activations.relu)(conv2_2)
    #40,40,384  -> 40,40,128

    conv1_1_up = layers.Conv2DTranspose(64,3,strides=1,padding="same",activation=keras.activations.relu)(conv2_1)
    #40,40,128  -> 40,40,64
    conv1_1_up = layers.UpSampling2D(2)(conv1_1_up)
    #40,40,64   -> 80,80,64
    conv1_1 = layers.Concatenate(axis=3)([temporary1_0,conv1_1_up])
    #80,80,64 + 80,80,64 -> 80,80,128
    conv1_1 = layers.Conv2D(64,3,strides=1,padding="same",activation=keras.activations.relu)(conv1_1)
    #80,80,128  -> 80,80,64

    conv1_2_up = layers.Conv2DTranspose(64,3,strides=1,padding="same",activation=keras.activations.relu)(conv2_2)
    #40,40,128  -> 40,40,64
    conv1_2_up = layers.UpSampling2D(2)(conv1_2_up)
    #40,40,64   -> 80,80,64
    conv1_2 = layers.Concatenate(axis=3)([temporary1_0,conv1_1,conv1_2_up])
    #80,80,64 * 3  -> 80,80,192
    conv1_2 = layers.Conv2D(64,3,strides=1,padding="same",activation=keras.activations.relu)(conv1_2)
    #80,80,192  -> 80,80,64

    conv0_1_up = layers.Conv2DTranspose(32,3,strides=1,padding="same",activation=keras.activations.relu)(conv1_1)
    #80,80,64   -> 80,80,32
    conv0_1_up = layers.UpSampling2D(2)(conv0_1_up)
    #80,80,32   -> 160,160,32
    conv0_1 = layers.Concatenate(axis=3)([temporary0_0,conv0_1_up])
    #160,160,32 -> 160,160,64
    conv0_1 = layers.Conv2D(32,3,strides=1,padding="same",activation=keras.activations.relu)(conv0_1)
    #160,160,64 -> 160,160,32

    conv0_2_up = layers.Conv2DTranspose(32,3,strides=1,padding="same",activation=keras.activations.relu)(conv1_2)
    #80,80,64   -> 80,80,32
    conv0_2_up = layers.UpSampling2D(2)(conv0_2_up)
    #80,80,32   -> 160,160,32
    conv0_2 = layers.Concatenate(axis=-1)([temporary0_0,conv0_1,conv0_2_up])
    #160,160,32 * 3  -> 160,160,96
    conv0_2 = layers.Conv2D(32,3,strides=1,padding="same",activation=keras.activations.relu)(conv0_2)
    #160,160,96 -> 160,160,32

    output1 = layers.Conv2D(num_classes,1,activation=keras.activations.sigmoid)(conv0_1)
    output2 = layers.Conv2D(num_classes,1,activation=keras.activations.sigmoid)(conv0_2)
    #160,160,32 -> 160,160,3


    model = keras.Model(inputs,[output1,output2])

    return model

keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()




train_gen = OxfordPets( batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = OxfordPets( batch_size, img_size, val_input_img_paths  , val_target_img_paths)

tbCallBack = TensorBoard(log_dir="static\\log")

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy",metrics=['accuracy'])


epochs = 15
# model.fit(train_gen, epochs=epochs,validation_data=val_gen)
model.fit(train_gen, epochs=epochs, validation_data=val_gen,callbacks=[tbCallBack])

model.save('oxford_unetpp.h5')
