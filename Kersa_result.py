import os


# 测试集
test_input_dir = "static/DataSet/test/images/"
test_target_dir = "static/DataSet/test/annotations/trimaps/"

img_size = (160, 160)
num_classes = 3
batch_size = 2

# 测试
test_input_img_paths = sorted(
    [
        os.path.join(test_input_dir, fname)
        for fname in os.listdir(test_input_dir)
        if fname.endswith(".jpg")
    ]
)
test_target_img_paths = sorted(
    [
        os.path.join(test_target_dir, fname)
        for fname in os.listdir(test_target_dir)
        if fname.endswith(".png")
    ]
)


import PIL
from PIL import ImageOps



"""
## Prepare `Sequence` class to load & vectorize batches of data
"""

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
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




# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()


model = keras.models.load_model('oxford_unetpp.h5')





"""
## Visualize predictions
"""


test_gen  = OxfordPets( batch_size, img_size, test_input_img_paths , test_target_img_paths)
#
# ResNet
predict = model.predict(test_gen)
loss,loss_1,loss_2,acc_1,acc_2  = model.evaluate(test_gen)
print(str(loss_1)+'----'+str(acc_1))
print(str(loss_2)+'----'+str(acc_2))




def display_mask(i):
    mask_1 = np.argmax(predict[0][i], axis=-1)
    mask_2 = np.argmax(predict[1][i], axis=-1)
    mask_1 = np.expand_dims(mask_1, axis=-1)
    mask_2 = np.expand_dims(mask_2,axis=-1)
    #     [160,160,1]
    img_1 = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask_1))
    img_2 = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask_2))
    # display(img)
    return img_1,img_2





import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
i = random.randint(0,len(test_input_img_paths)-1)
def get_result(i):
    img1  = mpimg.imread(test_input_img_paths[i])
    img1_ = mpimg.imread(test_target_img_paths[i])
    img1__,img1___ = display_mask(i)


    title = 'loss_1:' + str(loss_1)[0:5] + '  acc_1:' + str(acc_1)[0:5] +'\nloss_2:'+str(loss_2)[0:5] + '  acc_2:'+str(acc_2)[0:5]
    fname = 'static/result/' + str(i) + '.jpg'


    # plt.figure("Image") # 图像窗口名称


    plt.subplot(2,2,1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Input')

    plt.subplot(2,2,2)
    plt.imshow(img1_)
    plt.axis('off')
    plt.title('Ground Truth')

    plt.subplot(2,2,3)
    plt.imshow(img1__)
    plt.axis('off')
    plt.title('Predict_1')

    plt.subplot(2,2,4)
    plt.imshow(img1___)
    plt.axis('off')
    plt.title('Predict_2')

    plt.suptitle(title, y=0.1)
    plt.savefig(fname=fname,figsize=[10,10])

if __name__ == '__main__':
    get_result(i)