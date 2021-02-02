from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self):
        pass

    def train(self, model, training_data, epochs=1, validation_data=None, save_path=None):
        batches_per_epoch = len(training_data)

        for epoch in range(epochs):
            loss = 0
            bar = tqdm(range(batches_per_epoch), desc='epoch ' + str(epoch), leave=True, unit='B')

            for batch in bar:
                x, y = training_data[batch]

                loss = 0.5 * (model.train_on_batch(x, y) + loss)

                bar.set_postfix(loss=loss)

            if save_path != None:
                model.save(save_path)
        
        return loss
