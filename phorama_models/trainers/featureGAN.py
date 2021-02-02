from .trainer import Trainer
from tqdm import tqdm

class FeatureGANTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def train(self, generator, discriminator, features, gan, training_data, epochs=1, validation_data=None, generator_save_path=None, discriminator_save_path=None):
        batches_per_epoch = len(training_data)

        for epoch in range(epochs):
            discriminator_loss = 0
            generator_loss = (0, 0, 0, 0)
            
            bar = tqdm(range(batches_per_epoch), desc='epoch ' + str(epoch), leave=True)

            for batch in bar:
                x, y = training_data[batch]

                fake_hr = generator.predict(x)

                valid = np.ones((train.batch_size, 1))
                fake = np.zeros((train.batch_size, 1))

                x_new = np.concatenate((y, fake_hr), axis=0)
                y_new = np.concatenate((valid, fake), axis=0)

                discriminator_loss = 0.5 * (discriminator.train_on_batch(x_new, y_new) + discriminator_loss)

                real_features = features.predict(y)
                tmp_generator_loss = gan.train_on_batch(x, [valid, real_features, y])
                generator_loss = (0.5*(tmp_generator_loss[0]+generator_loss[0]), 0.5*(tmp_generator_loss[1]+generator_loss[1]), 0.5*(tmp_generator_loss[2]+generator_loss[2]), 0.5*(tmp_generator_loss[3]+generator_loss[3]))

                a = str(discriminator_loss)[:6]
                b = str(generator_loss[0])[:6]
                c = str(generator_loss[1])[:6]
                d = str(generator_loss[2])[:6]
                e = str(generator_loss[3])[:6]

                bar.set_postfix(disc_loss=a, val_los=b, feat_loss=c, img_loss=d, gen_loss=e)
                
            if generator_save_path != None:
                generator.save(generator_save_path)

            if discriminator_save_path != None:
                discriminator.save(discriminator_save_path)
