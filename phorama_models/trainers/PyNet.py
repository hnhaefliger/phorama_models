from .trainer import Trainer
from tensorflow.keras.models import Model
from tqdm import tqdm
import numpy as np
from ..models.PyNet import Level1Loss, Level2_3Loss, Level4_5Loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2DTranspose

class PyNetTrainer:
    def train(self, pynet, training_data, epochs=1, validation_data=None, save_path=None):
        batches_per_epoch = len(training_data)

        print('training level 5')

        pynet.level5.model.trainable = True
        level5 = Sequential()
        level5.add(pynet.level5.model)
        level5.add(Conv2DTranspose(3, (2, 2), activation='sigmoid'))
        level5.compile(loss=Level4_5Loss(), optimizer='adam')

        for i in range(epochs):
            level5.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level5.model.trainable = False

        print('training level 4')

        pynet.level4.model.trainable = True
        level4 = Sequential()
        level4.add(pynet.level4.model)
        level4.add(Conv2DTranspose(3, (2, 2), activation='sigmoid'))
        level4.compile(loss=Level4_5Loss(), optimizer='adam')

        for i in range(epochs):
            level4.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level4.model.trainable = False

        print('training level 3')

        pynet.level3.model.trainable = True
        level3 = Sequential()
        level3.add(pynet.level3.model)
        level3.add(Conv2DTranspose(3, (2, 2), activation='sigmoid'))
        level3.compile(loss=Level2_3Loss(), optimizer='adam')

        for i in range(epochs):
            level3.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level3.model.trainable = False

        print('training level 2')

        pynet.level2.model.trainable = True
        level2 = Sequential()
        level2.add(pynet.level2.model)
        level2.add(Conv2DTranspose(3, (2, 2), activation='sigmoid'))
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
    
