from .trainer import Trainer
from tensorflow.keras.models import Model
from tqdm import tqdm
import numpy as np

class GANTrainer:
    def train(self, generator, discriminator, gan, training_data, epochs=1, validation_data=None, generator_save_path=None, discriminator_save_path=None):
        batches_per_epoch = len(training_data)

        for epoch in range(epochs):
            discriminator_loss = 0
            generator_loss = 0
            
            bar = tqdm(range(batches_per_epoch), desc='epoch ' + str(epoch), leave=True, unit='B')

            for batch in bar:
                x, y = training_data[batch]

                fake_y = generator.predict(x)

                valid = np.ones((training_data.batch_size, 1))
                fake = np.zeros((training_data.batch_size, 1))

                x_new = np.concatenate((y, fake_hr), axis=0)
                y_new = np.concatenate((valid, fake), axis=0)

                discriminator_loss = 0.5 * (discriminator.train_on_batch(x_new, y_new) + discriminator_loss)

                generator_loss = 0.5*(gan.train_on_batch(x, valid) + generator_loss)

                bar.set_postfix(disc_loss=discriminator_loss, gen_loss=generator_loss)

            if generator_save_path != None:
                generator.save(generator_save_path)

            if discriminator_save_path != None:
                discriminator.save(discriminator_save_path)
    
