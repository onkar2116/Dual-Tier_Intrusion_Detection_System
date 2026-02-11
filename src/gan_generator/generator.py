import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator network for producing synthetic attack traffic.

    Takes a latent vector z (random noise) and produces
    a feature vector that mimics real attack traffic.

    Architecture:
    - Linear(latent_dim -> 128) -> LeakyReLU
    - Linear(128 -> 256) -> BN -> LeakyReLU
    - Linear(256 -> 512) -> BN -> LeakyReLU
    - Linear(512 -> output_dim) -> Tanh
    """

    def __init__(self, latent_dim: int, output_dim: int, num_classes: int = 0):
        super(Generator, self).__init__()

        self.conditional = num_classes > 0
        input_dim = latent_dim + num_classes if self.conditional else latent_dim

        if self.conditional:
            self.class_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels=None):
        if self.conditional and labels is not None:
            label_emb = self.class_embedding(labels)
            z = torch.cat([z, label_emb], dim=1)
        return self.model(z)
