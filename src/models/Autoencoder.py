from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_length):
        super(Autoencoder, self).__init__()
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


    def forward(self, x, mask=None):
        x = self.encoder[0](x)
        if mask is not None:
            x = x * mask[:256] 

        x = self.encoder[1](x)

        x = self.encoder[2](x)
        if mask is not None:
            x = x * mask[256:384]  

        x = self.encoder[3](x)

        x = self.decoder[0](x)
        if mask is not None:
            x = x * mask[384:640]

        x = self.decoder[1](x)

        x = self.decoder[2](x)
        
        x = self.decoder[3](x)

        return x

    
    """
    Forward o aplicacion de mascara para unicamente la capa de cuello de botella. (128 neuronas)
    """
    # def forward(self, x, mask=None):
    #     x = self.encoder[0](x)

    #     x = self.encoder[1](x)

    #     x = self.encoder[2](x)
    #     if mask is not None:
    #         x = x * mask  

    #     x = self.encoder[3](x)

    #     x = self.decoder[0](x)

    #     x = self.decoder[1](x)

    #     x = self.decoder[2](x)
        
    #     x = self.decoder[3](x)

    #     return x
    
    