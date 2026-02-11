import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real vs fake traffic.

    Architecture:
    - Linear(input_dim -> 512) -> LeakyReLU -> Dropout
    - Linear(512 -> 256) -> LeakyReLU -> Dropout
    - Linear(256 -> 1)

    No Sigmoid at output (using WGAN-GP which needs raw scores).
    """

    def __init__(self, input_dim: int, num_classes: int = 0):
        super(Discriminator, self).__init__()

        self.conditional = num_classes > 0
        total_input = input_dim + num_classes if self.conditional else input_dim

        if self.conditional:
            self.class_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(total_input, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

    def forward(self, x, labels=None):
        if self.conditional and labels is not None:
            label_emb = self.class_embedding(labels)
            x = torch.cat([x, label_emb], dim=1)
        return self.model(x)
