import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, GlobalAveragePooling2D, Dense, Concatenate, LeakyReLU, BatchNormalization, Add, AveragePooling2D
from .model import PhoramaModel

def DoubleConv(inputs, filters, kernel, normalize=False, alpha=0.3):
    conv = inputs
    conv = Conv2D(filters, kernel, padding='same')(conv)

    if normalize:
        conv = BatchNormalization()(conv)

    conv = LeakyReLU(alpha)(conv)
    conv = Conv2D(filters, kernel, padding='same')(conv)

    if normalize:
        conv = BatchNormalization()(conv)

    conv = LeakyReLU(alpha)(conv)
    return conv


def MultiConv(inputs, filters, five=False, seven=False, nine=False, normalize=False, alpha=0.3):
    output = DoubleConv(inputs, filters, (3, 3), normalize=normalize, alpha=alpha)

    if five:
        inner = DoubleConv(inputs, filters, (5, 5), normalize=normalize, alpha=alpha)
        output = Concatenate()(output, inner)

    if seven:
        inner = DoubleConv(inputs, filters, (7, 7), normalize=normalize, alpha=alpha)
        output = Concatenate()(output, inner)

    if nine:
        inner = DoubleConv(inputs, filters, (9, 9), normalize=normalize, alpha=alpha)
        output = Concatenate()(output, inner)

    return output


def Final(inputs, filters, kernel):
    inner = inputs

    inner = Conv2D(filters, kernel, activation='tanh')(inner)

    return inner


class Level5(PhoramaModel):
    def __init__(self, load_path=None, optimizer='Adam'):
        inputs = Input(shape=(None, None, 3))
        inner = inputs

        inner = MultiConv(inner, 512, normalize=True)

        inner1 = MultiConv(inner, 512, normalize=True)
        inner = Add()(inner, inner1)

        inner1 = MultiConv(inner, 512, normalize=True)
        inner = Add()(inner, inner1)

        inner = MultiConv(inner, 512, normalize=True)

        outputs = inner

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)


class Level4(PhoramaModel):
    def __init__(self, load_path=None, load_level5=None, optimizer='Adam'):
        level5 = Level5(load_path=load_level5)
        level5.model.trainable = False

        inputs = Input(shape=(None, None, 3))

        inner = inputs

        inner = MultiConv(inner, 256, normalize=True)

        prev = AveragePooling2D((2,2))(inner)
        prev = level5(prev)

        inner = Concatenate()(inner, prev)

        inner = MultiConv(inner, 256, normalize=True)

        inner1 = MultiConv(inner, 256, normalize=True)
        inner = Add()(inner, inner1)

        inner1 = MultiConv(inner, 512, normalize=True)
        inner = Add()(inner, inner1)

        inner = MultiConv(inner, 512, normalize=True)

        outputs = inner

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)


class PyNet(PhoramaModel):
    def __init__(self, load_path=None, optimizer='Adam'):
        inputs = Input(shape=(None, None, 3))
        inner = inputs

        inner = Conv2D(3, (3,3), padding='same', activation='sigmoid')(inner)

        outputs = inner

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)


test = PyNet()
