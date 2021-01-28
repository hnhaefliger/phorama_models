import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

class PhoramaDiscriminator:
    def __init__(self, load_path):
        inputs = Input(shape=(None, None, 3))
        inner = inputs
        inner = Flatten()(inner)
        inner = Dense(1, activation='sigmoid')(inner)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='binary_crossentropy', optimizer='Adam')
        self.model.load_weights(load_path)

    def predict(self, data):
        return self.model.predict(data)
