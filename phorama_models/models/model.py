import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

class PhoramaModel:
    def __init__(self, load_path):
        inputs = Input(shape=(None, None, 3))
        inner = inputs
        inner = Conv2D(3, (3,3), padding='same')(inner)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer='Adam')
        self.model.load_weights(load_path)

    def predict(self, data):
        return self.model.predict(data)
