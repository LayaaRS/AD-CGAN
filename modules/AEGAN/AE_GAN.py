import torch
import torch.nn as nn
from .spectral_normalization import SpectralNorm

#  All images of (3, 32, 32)


class Discriminator_32(nn.Module):
    def __init__(self, latent_size, dropout=0.3, output_size=2):
        super(Discriminator_32, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2
        self.convs = nn.ModuleList()

        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, bias=True)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(256, 512, 4, stride=3, bias=True)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.final = nn.Sequential(
            SpectralNorm(nn.Linear(512, self.output_size, bias=True))
        )

    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
        output = self.convs[-1](output_pre)
        output = self.final(output.view(output.shape[0], -1))
        return output.squeeze(), output_pre


class Generator_32(nn.Module):
    def __init__(self, latent_size):
        super(Generator_32, self).__init__()
        self.latent_dim = latent_size
        self.leaky_value = 0.2
        self.deconv = nn.ModuleList()

        self.output_bias = nn.Parameter(
            torch.zeros(3, 32, 32), requires_grad=True)

        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 512, 4,
                               stride=2, bias=True, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=2, bias=True),
            nn.Tanh()
        ))

    def forward(self, noise):
        x = noise
        for layer in self.deconv:
            x = layer(x)
        return x


class Encoder_32(nn.Module):
    def __init__(self, latent_size, dropout=0.3, noise=False):
        super(Encoder_32, self).__init__()
        self.dropout = dropout
        self.latent_size = latent_size
        self.leaky_value = 0.2
        self.conv = nn.ModuleList()

        if noise:
            self.latent_size *= 2

        self.nonlinear = nn.LayerNorm(self.latent_size)
        # self.nonlinear = nn.Tanh()

        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, bias=True)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, bias=True)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(512, self.latent_size, 4,
                         stride=1, bias=True, padding=1)),
        ))

    def forward(self, x):
        output = x
        for layer in self.conv[:-1]:
            output = layer(output)
        output = self.conv[-1](output)
        output = self.nonlinear(output.view(output.shape[0], -1))
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        return output


#  All images of (1, 28, 28); MNIST, FashionMNIST

class Discriminator_28(nn.Module):
    def __init__(self, latent_size, dropout=0.2, output_size=2):
        super(Discriminator_28, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.1
        self.convs = nn.ModuleList()

        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(1, 64, 3, stride=2, bias=True)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, bias=True)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(128, 256, 3, stride=2, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))

        self.final1 = SpectralNorm(nn.Linear(6400, 1024, bias=True))
        self.final2 = SpectralNorm(
            nn.Linear(1024, self.output_size, bias=True))

    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
        output = self.convs[-1](output_pre)
        output = self.final1(output.view(output.shape[0], -1))
        output = self.final2(output)
        # output = torch.sigmoid(output)
        return output.squeeze(), output_pre


class Generator_28(nn.Module):
    def __init__(self, latent_size):
        super(Generator_28, self).__init__()
        self.latent_dim = latent_size
        self.leaky_value = 0.1
        self.deconv = nn.ModuleList()

        self.output_bias = nn.Parameter(
            torch.zeros(1, 28, 28), requires_grad=True)

        self.Linear = nn.Sequential(
            nn.Linear(self.latent_dim, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 7*7*128, bias=True),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(),
        )

        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(7*7*128, 64, 5, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        ))
        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(64, 64, 5, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        ))
        self.deconv.append(nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, bias=True),
            nn.Tanh(),
        ))

    def forward(self, noise):
        x = noise
        x = self.Linear(x.view(x.shape[0], -1))
        x = torch.unsqueeze(x, 2).unsqueeze(3)
        for layer in self.deconv:
            x = layer(x)
        return x


class Encoder_28(nn.Module):
    def __init__(self, latent_size, dropout=0.3, noise=False):
        super(Encoder_28, self).__init__()
        self.dropout = dropout
        self.latent_size = latent_size
        self.leaky_value = 0.1
        self.conv = nn.ModuleList()

        # if noise:
        #   self.latent_size *= 2

        self.nonlinear = nn.LayerNorm(self.latent_size)
        # self.nonlinear = nn.Tanh()

        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(1, 64, 3, stride=1, bias=True)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 128, 3, stride=2, bias=True)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 3, stride=2, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))

        self.linear = SpectralNorm(
            nn.Linear(6400, self.latent_size, bias=True))

    def forward(self, x):
        output = x
        for layer in self.conv:
            output = layer(output)

        output = self.linear(output.view(output.shape[0], -1))
        output = self.nonlinear(output)
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        return output

#  images of (3, 64, 64)


class Discriminator_64(nn.Module):
    def __init__(self, latent_size, input_size):
        super(Discriminator_64, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(self.input_size, 64, 3, 2, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(128, 256, 3, 2, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                SpectralNorm(nn.Conv2d(256, 256, 3, 2, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )

        self.final = nn.Sequential(SpectralNorm(
            nn.Linear(256 * 4 * 4, self.latent_size)), nn.Tanh())

    def forward(self, x):
        for layers in self.convs:
            output_pre = layers(x)
        output = self.final(output_pre.view(output_pre.shape[0], -1))
        return output, output_pre


class Generator_64(nn.Module):
    def __init__(self, latent_size, input_size):
        super(Generator_64, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size

        self.deconvs = nn.ModuleList()

        self.Linear = nn.Sequential(
            nn.Linear(self.latent_size, 64 * 4 * 4 *
                      4), nn.BatchNorm1d(64 * 4 * 4 * 4), nn.ReLU()
        )

        self.deconvs.append(nn.Sequential(nn.ConvTranspose2d(
            64 * 4, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU()))

        self.deconvs.append(nn.Sequential(nn.ConvTranspose2d(
            128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()))

        self.deconvs.append(nn.Sequential(nn.ConvTranspose2d(
            64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()))

        self.deconvs.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, self.input_size, 4, 2, 1),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        output = self.Linear(x.view(x.shape[0], -1))
        output = output.reshape(x.shape[0], 64 * 4, 4, 4)
        for layers in self.deconvs:
            output = layers(output)
        return output


class Encoder_64(nn.Module):
    def __init__(self, latent_size, dropout=0.3, noise=False):
        super(Encoder_64, self).__init__()
        self.dropout = dropout
        self.latent_size = latent_size
        self.leaky_value = 0.2
        self.conv = nn.ModuleList()

        if noise:
            self.latent_size *= 2

        self.nonlinear = nn.LayerNorm(self.latent_size)
        # self.nonlinear = nn.Tanh()

        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(3, 128, 5, stride=2, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(128, 256, 5, stride=2, bias=True)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(256, 512, 5, stride=2, bias=True)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(512, self.latent_size,
                             5, stride=1, bias=True)),
            )
        )

    def forward(self, x):
        output = x
        for layer in self.conv[:-1]:
            output = layer(output)
        output = self.conv[-1](output)
        output = self.nonlinear(output.view(output.shape[0], -1))
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        return output
