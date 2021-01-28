import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, LeakyReLU, Flatten
from .discriminator import PhoramaDiscriminator

def DBlock(inputs, filters, strides=1, bn=True):
    x = Conv2D(filters, (3,3), strides=strides, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    if bn:
        x = BatchNormalization(momentum=0.8)(x)
    
    return x

class SRGAN(PhoramaDiscriminator):
    def __init__(self, load_path=None, optimizer='Adam'):
        inputs = Input(shape=(64,64,3))
        inner = inputs

        inner = DBlock(inner, 64, bn=False)
        inner = DBlock(inner, 64, strides=2)

        filters = 128
        for _ in range(3):
            inner = DBlock(inner, filters)
            inner = DBlock(inner, filters, strides=2)
            filters = filters * 2

        inner = Flatten()(inner)

        inner = Dense(1024)(inner)
        inner = LeakyReLU(alpha=0.2)(inner)

        outputs = Dense(1, activation='sigmoid')(inner)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)
