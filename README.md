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
