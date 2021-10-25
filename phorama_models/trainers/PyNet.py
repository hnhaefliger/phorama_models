from .trainer import Trainer
from tensorflow.keras.models import Model
from tqdm import tqdm
import numpy as np
from ..models.PyNet import Level1Loss, Level2_3Loss, Level4_5Loss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose, Input

class PyNetTrainer:
    def train(self, pynet, training_data, epochs=1, validation_data=None, save_path=None):
        batches_per_epoch = len(training_data)

        print('training level 5')

        pynet.level5.model.trainable = True
        inputs = Input((None, None, 3))
        inner = pynet.level5.model(input)
        inner = Conv2DTranspose(3, (2, 2), activation='sigmoid')(inner)
        level5 = Model(inputs=inputs, outputs=inner)
        level5.compile(loss=Level4_5Loss(), optimizer='adam')

        for i in range(epochs):
            level5.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level5.model.trainable = False

        print('training level 4')

        pynet.level4.model.trainable = True
        inputs = Input((None, None, 3))
        inner = pynet.level4.model(input)
        inner = Conv2DTranspose(3, (2, 2), activation='sigmoid')(inner)
        level4 = Model(inputs=inputs, outputs=inner)
        level4.compile(loss=Level4_5Loss(), optimizer='adam')

        for i in range(epochs):
            level4.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level4.model.trainable = False

        print('training level 3')

        pynet.level3.model.trainable = True
        inputs = Input((None, None, 3))
        inner = pynet.level3.model(input)
        inner = Conv2DTranspose(3, (2, 2), activation='sigmoid')(inner)
        level3 = Model(inputs=inputs, outputs=inner)
        level3.compile(loss=Level2_3Loss(), optimizer='adam')

        for i in range(epochs):
            level3.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level3.model.trainable = False

        print('training level 2')

        pynet.level2.model.trainable = True
        inputs = Input((None, None, 3))
        inner = pynet.level2.model(input)
        inner = Conv2DTranspose(3, (2, 2), activation='sigmoid')(inner)
        level2 = Model(inputs=inputs, outputs=inner)
        level2.compile(loss=Level2_3Loss(), optimizer='adam')

        for i in range(epochs):
            level2.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level2.model.trainable = False

        print('training level 1')

        pynet.level1.model.trainable = True
        pynet.level1.model.compile(loss=Level1Loss(), optimizer='adam')

        for i in range(epochs):
            pynet.level1.model.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)
    
