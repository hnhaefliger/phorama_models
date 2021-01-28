import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import VGG19
from .features import PhoramaFeatures

class VGG(PhoramaFeatures):
    def __init__(self):
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=(64,64,3))
        vgg = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)

        inputs = Input(shape=(64,64,3))
        outputs = vgg(inputs)

        self.vgg = Model(inputs=inputs, outputs=outputs)
        self.vgg.compile(loss='mse', optimizer='Adam')
