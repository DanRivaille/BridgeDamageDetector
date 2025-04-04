from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_length: int,
                 layer_to_mask: str,
                 first_layer_size: int,
                 bottleneck_layer_size: int,
                 last_layer_size: int
                 ):
        super().__init__()
        self.layer_to_mask = layer_to_mask
        self.input_length = input_length
        self.encoder = nn.Sequential(
            nn.Linear(input_length, first_layer_size),
            nn.Tanh(),
            nn.Linear(first_layer_size, bottleneck_layer_size),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_layer_size, last_layer_size),
            nn.Tanh(),
            nn.Linear(last_layer_size, input_length),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
