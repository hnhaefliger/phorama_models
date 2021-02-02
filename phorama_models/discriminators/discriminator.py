import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

class PhoramaDiscriminator:
    def __init__(self, load_path=None, optimizer='Adam'):
        inputs = Input(shape=(None, None, 3))
        inner = inputs
        inner = Flatten()(inner)
        inner = Dense(1, activation='sigmoid')(inner)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)
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
