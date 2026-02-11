import os
import torch
import numpy as np
from src.gan_generator.generator import Generator
from src.gan_generator.discriminator import Discriminator
from src.gan_generator.train_gan import WGANGPTrainer


class WGANGP:
    """Wrapper combining Generator, Discriminator, and training logic."""

    def __init__(self, config, feature_dim, num_classes=0):
        self.config = config
        self.feature_dim = feature_dim
        self.latent_dim = config['gan']['latent_dim']
        self.num_classes = num_classes

        self.generator = Generator(self.latent_dim, feature_dim, num_classes)
        self.discriminator = Discriminator(feature_dim, num_classes)
        self.trainer = WGANGPTrainer(self.generator, self.discriminator, config)
        self.device = self.trainer.device

    def train(self, real_data, epochs=None):
        """Train the WGAN-GP model."""
        epochs = epochs or self.config['gan']['epochs']
        batch_size = self.config['gan']['batch_size']

        gen_losses, disc_losses = self.trainer.train(
            real_data, epochs=epochs, batch_size=batch_size
        )

        return {
            'gen_losses': gen_losses,
            'disc_losses': disc_losses,
            'epochs': epochs
        }

    def generate(self, n_samples, labels=None):
        """Generate synthetic attack traffic samples."""
        return self.trainer.generate_samples(n_samples, labels)

    def save(self, save_dir):
        """Save generator and discriminator weights."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(save_dir, 'generator_final.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, 'discriminator_final.pth'))

    def load(self, save_dir):
        """Load generator and discriminator weights."""
        self.generator.load_state_dict(
            torch.load(os.path.join(save_dir, 'generator_final.pth'), map_location=self.device)
        )
        self.discriminator.load_state_dict(
            torch.load(os.path.join(save_dir, 'discriminator_final.pth'), map_location=self.device)
        )
        self.generator.to(self.device)
        self.discriminator.to(self.device)
