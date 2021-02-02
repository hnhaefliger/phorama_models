from .GAN import PhoramaGAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class FeatureGAN(PhoramaGAN):
    def __init__(self, generator, discriminator, features, optimizer='Adam'):
        inputs = Input(shape=(None,None,3))
        inner = inputs

        inner = generator(inner)

        discriminator.setTrainable(False)
        validity = discriminator(inner)

        features.setTrainable(False)
        features = features(inner)

        self.model = Model(inputs=inputs, outputs=[validity, features, inner])
        self.model.compile(loss=['binary_crossentropy', 'mse', 'mse'], loss_weights=[1e-3, 1e0, 6e-1], optimizer=optimizer)
