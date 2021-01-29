from phorama_models import models
from phorama_models import discriminators
from phorama_models import features
from phorama_models import trainers
from phorama_models import utils

test1 = models.SRGAN()
test2 = discriminators.SRGAN()
#test3 = features.VGG()
test4 = models.RSGUNet()
test1.model.save('netron.h5')
