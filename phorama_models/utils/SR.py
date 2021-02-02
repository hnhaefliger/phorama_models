from .sequence import ImageSequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.image import resize
import numpy as np
import random

class SRImageSequence(ImageSequence):
    def __init__(self, images, batch_size, low_res=(32,32), upscaling=2):
        self.images = images
        self.batch_size = batch_size
        self.low_res = (32,32)
        self.high_res = (low_res[0]*upscaling,low_res[1]*upscaling)
        self.upscaling = upscaling

    def __getitem__(self, idx):
        images = self.images[idx*self.batch_size:(idx+1)*self.batch_size]
        images = [img_to_array(load_img(img)) for img in images]

        x = [img.shape[0] - self.high_res[0] for img in images]
        y = [img.shape[1] - self.high_res[1] for img in images]

        x = [random.randint(0, i) if i >= 0 else -1 for i in x]
        y = [random.randint(0, i) if i >= 0 else -1 for i in y]

        images = [[line[y[i]:y[i]+self.high_res[1]] for line in images[i][x[i]:x[i]+self.high_res[0]]] if (y[i] >= 0 and x[i] >= 0) else np.random.rand(self.high_res[0],self.high_res[1],3)*255 for i in range(self.batch_size)]
        
        x = np.array(images)
        x = np.array(resize(x, self.low_res)) / 255
        y = np.array(images) / 255

        return x, y
