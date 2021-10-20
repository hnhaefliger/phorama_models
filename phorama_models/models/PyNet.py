import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, BatchNormalization, Add, AveragePooling2D
from tensorflow.keras.losses import Loss, mean_squared_error
from tensorflow.keras.applications import VGG19
from .model import PhoramaModel


def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


class Level1Loss(Loss):
    def __init__(self):
        super().__init__()
        self.vgg = VGG19(weights='imagenet', include_top=False)

    def __call__(self, y_true, y_pred):
        vgg_true = self.vgg.predict(y_true)
        vgg_pred = self.vgg.predict(y_pred)

        return 0.8 * mean_squared_error(vgg_true, vgg_pred) + 0.2 * mean_squared_error(y_true, y_pred)


class Level2_3Loss(Loss):
    def __init__(self):
        super().__init__()
        self.vgg = VGG19(weights='imagenet', include_top=False)

    def __call__(self, y_true, y_pred):
        vgg_true = self.vgg.predict(y_true)
        vgg_pred = self.vgg.predict(y_pred)

        return mean_squared_error(vgg_true, vgg_pred) + 0.05 * mean_squared_error(y_true, y_pred) + 0.75 * ssim_loss(y_true, y_pred)


class Level4_5Loss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)


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
        output = Concatenate()([output, inner])

    if seven:
        inner = DoubleConv(inputs, filters, (7, 7), normalize=normalize, alpha=alpha)
        output = Concatenate()([output, inner])

    if nine:
        inner = DoubleConv(inputs, filters, (9, 9), normalize=normalize, alpha=alpha)
        output = Concatenate()([output, inner])

    return output


def Final(inputs, filters, kernel):
    inner = inputs

    inner = Conv2D(filters, kernel, activation='tanh')(inner)

    return inner


class Level5(PhoramaModel):
    def __init__(self, load_path=None, optimizer='Adam'):
        inputs = Input(shape=(None, None, 256))
        inner = inputs

        inner = MultiConv(inner, 512, normalize=True)

        inner1 = MultiConv(inner, 512, normalize=True)
        inner = Add()([inner, inner1])

        inner1 = MultiConv(inner, 512, normalize=True)
        inner = Add()([inner, inner1])

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

        inputs = Input(shape=(None, None, 128))

        inner = inputs

        inner = MultiConv(inner, 256, normalize=True)

        prev = AveragePooling2D((2,2))(inner)
        prev = level5(prev)
        prev = Conv2DTranspose(256, (2,2))(prev)

        inner = Concatenate()([inner, prev])

        inner = MultiConv(inner, 256, normalize=True)
        inner1 = MultiConv(inner, 256, normalize=True)
        inner = Add()([inner, inner1])

        inner1 = MultiConv(inner, 256, normalize=True)
        inner = Add()([inner, inner1])

<<<<<<< HEAD
        inner = MultiConv(inner, 256, normalize=True)

=======
        inner1 = MultiConv(inner, 256, normalize=True)
        inner = Add()([inner, inner1])

        inner = MultiConv(inner, 256, normalize=True)

>>>>>>> 7df3623dfd002524f7c9e5c9d6e2fba766708998
        inner = Concatenate()([inner, prev])

        inner = MultiConv(inner, 256, normalize=True)

        outputs = inner

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)


<<<<<<< HEAD
class Level3(PhoramaModel):
    def __init__(self, load_path=None, load_level4=None, optimizer='Adam'):
        level4 = Level4(load_path=load_level4)
        level4.model.trainable = False

        inputs = Input(shape=(None, None, 64))

        inner = inputs

        inner_s = MultiConv(inner, 128, normalize=True)

        prev = AveragePooling2D((2, 2))(inner_s)
        prev = level4(prev)
        prev = Conv2DTranspose(128, (2, 2))(prev)

        inner = Concatenate()([inner_s, prev])

        inner1 = MultiConv(inner, 128, five=True, normalize=True)
        inner = Add()([inner, inner1])

        inner1 = MultiConv(inner, 128, five=True, normalize=True)
        inner = Add()([inner, inner1])

        inner1 = MultiConv(inner, 128, five=True, normalize=True)
        inner = Add()([inner, inner1])

        inner = MultiConv(inner, 128, five=True, normalize=True)

        inner = Concatenate()([inner, prev, inner_s])

        inner = MultiConv(inner, 128, normalize=True)

        outputs = inner

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)

        
class Level2(PhoramaModel):
    def __init__(self, load_path=None, load_level3=None, optimizer='Adam'):
        level3 = Level3(load_path=load_level3)
        level3.model.trainable = False

        inputs = Input(shape=(None, None, 32))

        inner = inputs

        inner_s = MultiConv(inner, 64, normalize=True)

        prev = AveragePooling2D((2, 2))(inner_s)
        prev = level3(prev)
        prev = Conv2DTranspose(64, (2, 2))(prev)

        inner = Concatenate()([inner_s, prev])

        inner1 = MultiConv(inner, 64, five=True, normalize=True)
        inner = Concatenate()([inner_s, inner1])

        inner1 = MultiConv(inner, 64, five=True, seven=True, normalize=True)
        inner = Add()([inner, inner1])

        inner1 = MultiConv(inner, 64, five=True, seven=True, normalize=True)
        inner = Add()([inner, inner1])

        inner1 = MultiConv(inner, 64, five=True, seven=True, normalize=True)
        inner = Add()([inner, inner1])

        inner1 = MultiConv(inner, 64, five=True, seven=True, normalize=True)
        inner = Concatenate()([inner_s, inner1])

        inner = MultiConv(inner, 64, five=True, normalize=True)

        inner = Concatenate()([inner, prev])

        inner = MultiConv(inner, 64, normalize=True)

        outputs = inner

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)


class Level1(PhoramaModel):
    def __init__(self, load_path=None, load_level2=None, optimizer='Adam'):
        level2 = Level2(load_path=load_level2)
        level2.model.trainable = False

        inputs = Input(shape=(None, None, 3))

        inner = inputs

        inner_s = MultiConv(inner, 32, normalize=False)

        prev = AveragePooling2D((2, 2))(inner_s)
        prev = level2(prev)
        prev = Conv2DTranspose(32, (2, 2))(prev)

        inner = Concatenate()([inner_s, prev])

        inner1 = MultiConv(inner, 32, five=True, normalize=False)
        inner = Concatenate()([inner_s, inner1])

        inner = MultiConv(inner, 32, five=True, seven=True, normalize=False)
        inner = MultiConv(inner, 32, five=True, seven=True, nine=True, normalize=False)

        inner1 = MultiConv(inner, 32, five=True, seven=True, nine=True, normalize=False)
        inner = Add()([inner, inner1])

        inner1 = MultiConv(inner, 32, five=True, seven=True, nine=True, normalize=False)
        inner = Add()([inner, inner1])

        inner1 = MultiConv(inner, 32, five=True, seven=True, nine=True, normalize=False)
        inner = Add()([inner, inner1])

        inner = MultiConv(inner, 32, five=True, seven=True, normalize=False)
        inner = Concatenate()([inner_s, inner])

        inner = MultiConv(inner, 32, five=True, normalize=False)
        inner = Concatenate()([inner_s, inner, prev])

        inner = MultiConv(inner, 32, normalize=False)
=======
class PyNet(PhoramaModel):
    def __init__(self, load_path=None, optimizer='Adam'):
        inputs = Input(shape=(None, None, 128))
        inner = inputs

        inner = Level4()(inner)
>>>>>>> 7df3623dfd002524f7c9e5c9d6e2fba766708998

        outputs = inner

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)
<<<<<<< HEAD


class PyNet(PhoramaModel):
    def __init__(self, load_path=None, optimizer='Adam'):
        inputs = Input(shape=(None, None, 3))
        inner = inputs

        inner = Level1()(inner)

        outputs = inner

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=optimizer)

        if load_path != None:
            self.model.load_weights(load_path)
=======
>>>>>>> 7df3623dfd002524f7c9e5c9d6e2fba766708998
