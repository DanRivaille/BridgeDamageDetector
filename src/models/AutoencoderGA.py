from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_length, layer_to_mask: str):
        super().__init__()
        self.layer_to_mask = layer_to_mask
        self.input_length = input_length
        self.encoder = nn.Sequential(
            nn.Linear(input_length, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, input_length),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
