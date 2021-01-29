import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, GlobalAveragePooling2D, Dense, Concatenate
from .model import PhoramaModel

def DownsampleBlock(inputs, filters):
    conv = inputs
    conv = Conv2D(filters, (3,3), padding='same', activation='relu')(conv)
    conv = Conv2D(filters, (3,3), padding='same', activation='relu')(conv)
    inner = Conv2D(filters, (3,3), strides=(2,2), padding='same', activation='relu')(conv)
    return conv, inner

def UpsampleBlock(inputs1, inputs2, filters):
    inner = Conv2DTranspose(filters, (2,2), strides=(2,2), activation='relu')(inputs1)
    inner = Concatenate()([inner, inputs2])
    inner = Conv2D(filters, (3,3), padding='same', activation='relu')(inner)
    inner = Conv2D(filters, (3,3), padding='same', activation='relu')(inner)
    return inner

class RSGUNet(PhoramaModel):
    def __init__(self, load_path=None, optimizer='Adam'):
        inputs = Input(shape=(None, None, 3))
        inner = inputs

        inner = Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu')(inner)

        conv1, pool1 = DownsampleBlock(inner, 16)
        conv2, pool2 = DownsampleBlock(pool1, 32)
        conv3, pool3 = DownsampleBlock(pool2, 64)
        conv4, pool4 = DownsampleBlock(pool3, 128)

        pool5 = Conv2D(256, (3,3), padding='same', activation='relu')(pool4)

        global_average = GlobalAveragePooling2D()(pool5)
        global_average = Dense(128, activation='relu')(global_average)

        feature = tf.expand_dims(global_average, axis=-2)
        feature = tf.expand_dims(feature, axis=-2)

        ones = tf.zeros(shape=tf.shape(conv4))
        global_feature = ones + feature
        
        pool4 = Concatenate()([global_feature, conv4])
        pool4 = Conv2D(128, (3,3), padding='same', activation='relu')(pool4)
        pool4 = Conv2D(128, (3,3), padding='same', activation='relu')(pool4)

        pool3 = UpsampleBlock(pool4, conv3, 64)
        pool2 = UpsampleBlock(pool3, conv2, 32)
        pool1 = UpsampleBlock(pool2, conv1, 16)

        inner = tf.math.multiply(pool1, inner)
        inner = Conv2DTranspose(3, (2,2), strides=(2,2), activation='relu')(inner)
        inner = Conv2D(3, (3,3), padding='same', activation='sigmoid')(inner)

        outputs = inner

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)
