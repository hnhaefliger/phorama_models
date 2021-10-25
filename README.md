# phorama-models

These are the image superresolution model training scripts for phorama. I tried to make the GANs as modular as possible so that it would be easier to experiment with different models later on.

To get some common image datasets:

```python3
phorama_models.datasets.labelme_12_50k.get('download_here')
phorama_models.datasets.indoorscene.get('download_here')
phorama_models.datasets.visualgenome.get('download_here')
phorama_models.datasets.coco2017.get('download_here')
```

To use directory as a dataset:

```python3
SRGAN_data = phorama_models.utils.SRImageSequence(phorama_models.utils.ImageFinder().search('download_here'), 8)
```

Now we need to create a model:

```python3
SRGAN_generator = phorama_models.models.SRGAN()

# SRGAN optimizes for the discriminator and features.
SRGAN_discriminator = phorama_models.discriminators.SRGAN()
SRGAN_features = phorama_models.features.VGG()

SRGAN_combined = phorama_models.gan.FeatureGAN(SRGAN_generator, SRGAN_discriminator, SRGAN_features)
```

And then train the model:

```python3
SRGAN_trainer = phorama_models.trainers.FeatureGANTrainer()

SRGAN_trainer.train(SRGAN_generator, SRGAN_discriminator, SRGAN_features, SRGAN_combined, SRGAN_data, epochs=10)
```

We can also experiment with different training setups by changing only certain lines:

e.g. To train SRGAN on only the discriminator:

```python3
SRGAN_generator = phorama_models.models.SRGAN()

SRGAN_discriminator = phorama_models.discriminators.SRGAN()

SRGAN_combined = phorama_models.gan.PhoramaGAN(SRGAN_generator, SRGAN_discriminator)

SRGAN_trainer = phorama_models.trainers.GANTrainer()
```

Or to train it directly on the MSE with the images:

```python3
SRGAN_generator = phorama_models.models.RSGUNet()

SRGAN_trainer = phorama_models.trainers.Trainer()
```

***Models currently implemented***:

- [x] SRGAN (https://arxiv.org/pdf/1609.04802.pdf)

- [x] RSGUNet (https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Huang_Range_Scaling_Global_U-Net_for_Perceptual_Image_Enhancement_on_Mobile_ECCVW_2018_paper.pdf)

- [x] PyNet (https://arxiv.org/pdf/2002.05509.pdf)

***Models to be implemented***:

- [ ] UNet (https://arxiv.org/pdf/1505.04597.pdf)

- [ ] GVTNet (https://arxiv.org/pdf/2008.02340.pdf)

- [ ] DPED (https://arxiv.org/pdf/1704.02470.pdf)

- [ ] CURL (https://arxiv.org/pdf/1911.13175.pdf)

- [ ] FEQE (https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Vu_Fast_and_Efficient_Image_Quality_Enhancement_via_Desubpixel_Convolutional_Neural_ECCVW_2018_paper.pdf)
