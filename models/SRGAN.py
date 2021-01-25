import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Activation, Add, BatchNormalization
from phorama.models.model import PhoramaModel

def ResidualBlock(inputs, filters):
    x = Conv2D(filters, (3,3), strides=1, padding='same')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(filters, (3,3), strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Add()([x, inputs])
    return x

def DeConv2D(inputs):
    x = UpSampling2D(size=2)(inputs)
    x = Conv2D(256, (3,3), strides=1, padding='same')(x)
    x = Activation('relu')(x)
    return x

class SRGAN(PhoramaModel):
    def __init__(self, load_path):
        inputs = Input(shape=(None, None, 3))
        inner = inputs

        inner1 = Conv2D(64, (9,9), strides=1, padding='same')(inner)
        inner1 = Activation('relu')(inner1)

        inner = inner1
        inner = ResidualBlock(inner, 64)

        for _ in range(15):
            inner = ResidualBlock(inner, 64)

        inner = Conv2D(64, (3,3), strides=1, padding='same')(inner)
        inner = BatchNormalization(momentum=0.8)(inner)
        inner = Add()([inner, inner1])
        inner = DeConv2D(inner)
        outputs = Conv2D(3, (9,9), strides=1, padding='same', activation='sigmoid')(inner)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer='Adam')
        self.model.load_weights(load_path)
