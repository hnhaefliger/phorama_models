from .trainer import Trainer
from tensorflow.keras.models import Model
from tqdm import tqdm
import numpy as np
from phorama_models.models.PyNet import Level1Loss, Level2_3Loss, Level4_5Loss

class PyNetTrainer:
    def train(self, pynet, training_data, epochs=1, validation_data=None, save_path=None):
        batches_per_epoch = len(training_data)

        print('training level 5')

        pynet.level5.trainable = True
        pynet.level5.compile(loss=Level4_5Loss(), optimizer='adam')

        for i in range(epochs):
            pynet.level5.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level5.trainable = False

        print('training level 4')

        pynet.level4.trainable = True
        pynet.level4.compile(loss=Level4_5Loss(), optimizer='adam')

        for i in range(epochs):
            pynet.level4.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level4.trainable = False

        print('training level 3')

        pynet.level3.trainable = True
        pynet.level3.compile(loss=Level2_3Loss(), optimizer='adam')

        for i in range(epochs):
            pynet.level3.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level3.trainable = False

        print('training level 2')

        pynet.level2.trainable = True
        pynet.level2.compile(loss=Level2_3Loss(), optimizer='adam')

        for i in range(epochs):
            pynet.level2.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)

        pynet.level3.trainable = False

        print('training level 1')

        pynet.level1.trainable = True
        pynet.level1.compile(loss=Level1Loss(), optimizer='adam')

        for i in range(epochs):
            pynet.level1.train(training_data, epochs=1, steps_per_epoch=batches_per_epoch)
            pynet.model.save(save_path)
    
