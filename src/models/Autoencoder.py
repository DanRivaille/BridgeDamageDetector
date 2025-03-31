from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_length, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, input_length),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x