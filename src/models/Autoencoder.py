from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_length, 32),
            nn.Tanh(),
            nn.Linear(32, 16),  
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),  
            nn.Tanh(),
            nn.Linear(32, input_length),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x