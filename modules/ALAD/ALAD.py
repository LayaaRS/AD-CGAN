import torch
import torch.nn as nn
from .spectral_normalization import SpectralNorm

#  All images of (1, 28, 28)


class Generator_MNIST(nn.Module):

    def __init__(self, latent_size):
        super(Generator_MNIST, self).__init__()
        self.latent_size = latent_size

        self.output_bias = nn.Parameter(
            torch.zeros(1, 28, 28), requires_grad=True)
        self.Linear = nn.Sequential(
            nn.Linear(self.latent_size, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 7*7*128, bias=True),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(),
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(7*7*128, 64, 5, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, 5, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 4, stride=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.Linear(input.view(input.shape[0], -1))
        output = torch.unsqueeze(output, 2).unsqueeze(3)
        output = self.main(output)
        return output


class Encoder_MNIST(nn.Module):

    def __init__(self, latent_size, noise=False):
        super(Encoder_MNIST, self).__init__()
        self.latent_size = latent_size

        if noise:
            self.latent_size *= 2
        self.main1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(1, 32, 3, stride=1, bias=False)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
        )
        self.main2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(32, 64, 3, stride=2, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.main3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 128, 3, stride=2, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )

        self.main4 = nn.Sequential(
            SpectralNorm(nn.Linear(3200, self.latent_size, bias=True))
        )

    def forward(self, input):
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3.view(x3.shape[0], -1))
        output = output.unsqueeze(2).unsqueeze(3)
        return output


class Discriminator_xz_MNIST(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xz_MNIST, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.infer_x = nn.Sequential(
            SpectralNorm(nn.Conv2d(1, 64, 4, stride=2, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.infer_z = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.latent_size,
                         512, 1, stride=1, bias=False)),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),
        )

        self.infer_joint = nn.Sequential(
            SpectralNorm(nn.Linear(2112, 1024, bias=True)),
            nn.LeakyReLU(inplace=True),
        )

        self.logit = SpectralNorm(nn.Linear(1024, 1, bias=True))

    def forward(self, x, z):
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        itermediate_layer = self.infer_joint(torch.cat([output_x.view(
            output_x.shape[0], -1), output_z.view(output_z.shape[0], -1)], dim=1))
        logit = self.logit(itermediate_layer)
        return logit.squeeze(), itermediate_layer


class Discriminator_xx_MNIST(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xx_MNIST, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_xx_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(1, 64, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )

        self.joint_xx_linear = nn.Sequential(
            SpectralNorm(nn.Linear(2048, 1))
        )

    def forward(self, x, x_prime):
        x = torch.cat((x, x_prime), 0)
        itermediate_layer = self.joint_xx_conv(x)
        logit = self.joint_xx_linear(
            itermediate_layer.view(itermediate_layer.shape[0], -1))
        return logit.squeeze(), itermediate_layer.squeeze()


class Discriminator_zz_MNIST(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_zz_MNIST, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_zz = nn.Sequential(
            SpectralNorm(nn.Linear(self.latent_size, 64)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Linear(64, 32)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )
        self.logit = nn.Sequential(
            SpectralNorm(nn.Linear(32, 1)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )

    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        itermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(itermediate_layer)
        return logit.squeeze(), itermediate_layer


##########################################################################

#  All images of (3, 32, 32)

class Generator_CIFAR10(nn.Module):

    def __init__(self, latent_size):
        super(Generator_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.leaky_value = 0.2

        self.output_bias = nn.Parameter(
            torch.zeros(3, 32, 32), requires_grad=True)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, 512, 4,
                               stride=2, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 5, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 3, 4, stride=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Encoder_CIFAR10(nn.Module):

    def __init__(self, latent_size, noise=False, do_spectral_norm=True):
        super(Encoder_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.leaky_value = 0.2

        if noise:
            self.latent_size *= 2
        self.main1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )
        self.main2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )

        self.main3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )

        self.main4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(512, self.latent_size, 4,
                         stride=1, bias=True, padding=1)),
            # nn.BatchNorm2d(self.latent_size)
        )

    def forward(self, input):
        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3)
        return output


class Discriminator_xz_CIFAR10(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xz_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.infer_x = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(256, 512, 4, stride=3, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )

        self.infer_z = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.latent_size,
                         512, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),

            SpectralNorm(nn.Conv2d(512, 512, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            SpectralNorm(nn.Conv2d(1024, 1024, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )
        self.logit = nn.Sequential(
            SpectralNorm(nn.Conv2d(1024, 1, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )

    def forward(self, x, z):
        x = x
        z = z
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        itermediate_layer = self.infer_joint(
            torch.cat([output_x, output_z], dim=1))
        logit = self.logit(itermediate_layer)
        return logit.squeeze(), itermediate_layer


class Discriminator_xx_CIFAR10(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xx_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_xx_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )

        self.joint_xx_linear = nn.Sequential(
            SpectralNorm(nn.Linear(3200, 1))
        )

    def forward(self, x, x_prime):
        x = torch.cat((x, x_prime), 0)
        itermediate_layer = self.joint_xx_conv(x)
        logit = self.joint_xx_linear(
            itermediate_layer.view(itermediate_layer.shape[0], -1))
        return logit.squeeze(), itermediate_layer.squeeze()


class Discriminator_zz_CIFAR10(nn.Module):

    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_zz_CIFAR10, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_zz = nn.Sequential(
            SpectralNorm(nn.Linear(self.latent_size, 64)),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Linear(64, 32)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )
        self.logit = nn.Sequential(
            SpectralNorm(nn.Linear(32, 1)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        )

    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        itermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(itermediate_layer)
        return logit.squeeze(), itermediate_layer


##########################################################################

#  images of the size (3, 64, 64)

class Discriminator_CD(nn.Module):
    def __init__(self, latent_size, input_size):
        super(Discriminator_CD, self).__init__()
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
            output = layers(x)
        output = self.final(output.view(output.shape[0], -1))
        return output


class Generator_CD(nn.Module):
    def __init__(self, latent_size, input_size):
        super(Generator_CD, self).__init__()
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


class Encoder_CD(nn.Module):
    def __init__(self, latent_size, dropout=0.3, noise=False):
        super(Encoder_CD, self).__init__()
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


class Discriminator_xz_CD(nn.Module):
    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xz_CD, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.infer_x = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            SpectralNorm(nn.Conv2d(256, 512, 4, stride=3, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )

        self.infer_z = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.latent_size,
                         512, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
            SpectralNorm(nn.Conv2d(512, 512, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(p=self.dropout),
        )

        self.infer_joint = nn.Sequential(
            SpectralNorm(nn.Conv2d(1024, 1024, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )
        self.logit = nn.Sequential(
            SpectralNorm(nn.Conv2d(1024, 1, 1, stride=1, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )

    def forward(self, x, z):
        x = x
        z = z
        output_x = self.infer_x(x)
        output_z = self.infer_z(z)
        intermediate_layer = self.infer_joint(
            torch.cat([output_x, output_z], dim=1))
        logit = self.logit(intermediate_layer)
        return logit.squeeze(), intermediate_layer


class Discriminator_xx_CD(nn.Module):
    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_xx_CD, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_xx_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            SpectralNorm(nn.Conv2d(128, 128, 5, stride=2, bias=False)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )

        self.joint_xx_linear = nn.Sequential(SpectralNorm(nn.Linear(3200, 1)))

    def forward(self, x, x_prime):
        x = torch.cat((x, x_prime), 0)
        intermediate_layer = self.joint_xx_conv(x)
        logit = self.joint_xx_linear(
            intermediate_layer.view(intermediate_layer.shape[0], -1))
        return logit.squeeze(), intermediate_layer.squeeze()


class Discriminator_zz_CD(nn.Module):
    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_zz_CD, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2

        self.joint_zz = nn.Sequential(
            SpectralNorm(nn.Linear(self.latent_size, 64)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            SpectralNorm(nn.Linear(64, 32)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )
        self.logit = nn.Sequential(SpectralNorm(
            nn.Linear(32, 1)), nn.LeakyReLU(self.leaky_value, inplace=True))

    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        intermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(intermediate_layer)
        return logit.squeeze(), intermediate_layer
