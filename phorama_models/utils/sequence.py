from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

class ImageSequence(Sequence):
    def __init__(self, images, batch_size):
        self.images = images
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.images) / self.batch_size)

    def __getitem__(self, idx):
        images = self.images[idx*self.batch_size:(idx+1)*self.batch_size]
        images = np.array([img_to_array(load_img(img)) for img in images])

        x = np.array(images) / 255
        y = np.array(images) / 255

        return x, y
