from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_length, layer_to_mask: str):
        super().__init__()
        self.layer_to_mask = layer_to_mask
        self.input_length = input_length
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
