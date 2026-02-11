import pytest
import torch
import numpy as np
from src.gan_generator.generator import Generator
from src.gan_generator.discriminator import Discriminator


def test_generator_output_shape():
    gen = Generator(latent_dim=100, output_dim=35)
    z = torch.randn(10, 100)
    output = gen(z)
    assert output.shape == (10, 35)


def test_discriminator_output_shape():
    disc = Discriminator(input_dim=35)
    x = torch.randn(10, 35)
    output = disc(x)
    assert output.shape == (10, 1)


def test_conditional_generator():
    gen = Generator(latent_dim=100, output_dim=35, num_classes=5)
    z = torch.randn(10, 100)
    labels = torch.randint(0, 5, (10,))
    output = gen(z, labels)
    assert output.shape == (10, 35)


def test_conditional_discriminator():
    disc = Discriminator(input_dim=35, num_classes=5)
    x = torch.randn(10, 35)
    labels = torch.randint(0, 5, (10,))
    output = disc(x, labels)
    assert output.shape == (10, 1)


def test_generator_tanh_range():
    gen = Generator(latent_dim=100, output_dim=35)
    z = torch.randn(100, 100)
    output = gen(z)
    assert output.min() >= -1.0
    assert output.max() <= 1.0


def test_gradient_penalty(config):
    from src.gan_generator.train_gan import WGANGPTrainer
    gen = Generator(latent_dim=100, output_dim=35)
    disc = Discriminator(input_dim=35)
    trainer = WGANGPTrainer(gen, disc, config)

    real = torch.randn(8, 35).to(trainer.device)
    fake = torch.randn(8, 35).to(trainer.device)
    gp = trainer.gradient_penalty(real, fake)
    assert gp.shape == ()  # Scalar
    assert gp.item() >= 0


def test_gan_short_training(config):
    """Test GAN training runs for 2 epochs without error."""
    from src.gan_generator.gan_model import WGANGP

    data = np.random.randn(100, 35).astype(np.float32)
    gan = WGANGP(config, feature_dim=35)
    result = gan.train(data, epochs=2)

    assert len(result['gen_losses']) == 2
    assert len(result['disc_losses']) == 2


def test_sample_generation(config):
    from src.gan_generator.gan_model import WGANGP

    gan = WGANGP(config, feature_dim=35)
    samples = gan.generate(20)
    assert samples.shape == (20, 35)
