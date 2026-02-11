import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.utils.config import resolve_path


class WGANGPTrainer:
    """
    Wasserstein GAN with Gradient Penalty trainer.

    WGAN-GP is more stable than vanilla GAN because:
    - Uses Wasserstein distance instead of JS divergence
    - Gradient penalty enforces Lipschitz constraint
    - No mode collapse issues
    """

    def __init__(self, generator, discriminator, config):
        self.gen = generator
        self.disc = discriminator
        self.config = config
        self.latent_dim = config['gan']['latent_dim']
        self.gp_weight = config['gan']['gradient_penalty_weight']

        self.gen_optimizer = optim.Adam(
            self.gen.parameters(),
            lr=config['gan']['learning_rate'],
            betas=(config['gan']['beta1'], config['gan']['beta2'])
        )
        self.disc_optimizer = optim.Adam(
            self.disc.parameters(),
            lr=config['gan']['learning_rate'],
            betas=(config['gan']['beta1'], config['gan']['beta2'])
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen.to(self.device)
        self.disc.to(self.device)

    def gradient_penalty(self, real_data, fake_data):
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        disc_interpolated = self.disc(interpolated)

        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty

    def train(self, real_data, epochs, batch_size, n_critic=5):
        """Train WGAN-GP."""
        dataset = TensorDataset(torch.FloatTensor(real_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        gen_losses = []
        disc_losses = []

        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            batches = 0

            for batch in dataloader:
                real_batch = batch[0].to(self.device)
                batch_size_actual = real_batch.size(0)

                # --- Train Discriminator ---
                for _ in range(n_critic):
                    self.disc_optimizer.zero_grad()

                    z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                    fake_batch = self.gen(z).detach()

                    disc_real = self.disc(real_batch).mean()
                    disc_fake = self.disc(fake_batch).mean()

                    gp = self.gradient_penalty(real_batch, fake_batch)
                    disc_loss = disc_fake - disc_real + self.gp_weight * gp
                    disc_loss.backward()
                    self.disc_optimizer.step()

                # --- Train Generator ---
                self.gen_optimizer.zero_grad()

                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                fake_batch = self.gen(z)
                gen_loss = -self.disc(fake_batch).mean()
                gen_loss.backward()
                self.gen_optimizer.step()

                epoch_gen_loss += gen_loss.item()
                epoch_disc_loss += disc_loss.item()
                batches += 1

            if batches == 0:
                continue

            avg_gen = epoch_gen_loss / batches
            avg_disc = epoch_disc_loss / batches
            gen_losses.append(avg_gen)
            disc_losses.append(avg_disc)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} | D Loss: {avg_disc:.4f} | G Loss: {avg_gen:.4f}")

            # Save checkpoint
            save_interval = self.config['gan']['save_interval']
            if epoch > 0 and epoch % save_interval == 0:
                save_dir = resolve_path('models/gan')
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.gen.state_dict(), os.path.join(save_dir, f'generator_epoch_{epoch}.pth'))

        return gen_losses, disc_losses

    def generate_samples(self, num_samples, labels=None):
        """Generate synthetic attack traffic samples."""
        self.gen.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            if labels is not None:
                labels = torch.LongTensor(labels).to(self.device)
            generated = self.gen(z, labels)
        return generated.cpu().numpy()
