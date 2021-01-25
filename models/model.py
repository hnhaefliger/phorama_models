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

    def preprocess(self, data):
        data = data / 255
        return data

    def postprocess(self, data):
        data = data * 255
        data = data.astype(np.uint8)
        return data

    def predict(self, data):
        data = self.preprocess(data)
        data = np.expand_dims(data, axis=0)
        data = self.model.predict(data)[0]
        data = np.array(data)
        data = self.postprocess(data)
        return data
