from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class PhoramaGAN:
    def __init__(self, generator, discriminator, optimizer='Adam'):
        inputs = Input(shape=(None,None,3))

        inner = generator(inputs)

        discriminator.setTrainable(False)
        inner = discriminator(inner)

        self.model = Model(inputs=inputs, outputs=inner)

        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)
        
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

