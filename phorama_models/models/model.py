import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

class PhoramaModel:
    def __init__(self, load_path=None, optimizer='Adam'):
        inputs = Input(shape=(None, None, 3))
        inner = inputs
        inner = Conv2D(3, (3,3), padding='same')(inner)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)

    def predict(self, data):
        return self.model.predict(data)

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def setTrainable(self, trainable):
        self.model.trainable = trainable

    def __call__(self, x):
        return self.model(x)

    def save(self, path):
        return self.model.save(path)

