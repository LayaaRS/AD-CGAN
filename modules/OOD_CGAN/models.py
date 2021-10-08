import torch
import torch.nn as nn
from .spectral_normalization import SpectralNorm


class Discriminator(nn.Module):
    """ Disciminator works on images of size (3, 32, 32).
    It uses spectral normalization. """
    def __init__(self, latent_size, dropout=0.3, output_size=1):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2
        self.convs = nn.ModuleList()

        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 5, stride=1, bias=True, padding=1)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),

            SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, bias=True)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 5, stride=1, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            # nn.Dropout2d(self.dropout),

            SpectralNorm(nn.Conv2d(256, 512, 5, stride=1, bias=True)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),
        ))
        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(512, 1024, 3, stride=1, bias=True)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(1024, 2048, 3, stride=2, bias=True)),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),
        ))

        self.local = SpectralNorm(nn.Conv2d(512, 2048, 1, 1))
        self.final = nn.Sequential(
            SpectralNorm(nn.Linear(2048, 2048, bias=True)),
            nn.ReLU(),
            SpectralNorm(nn.Linear(2048, self.output_size, bias=True)),
            )

    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
        output_local = self.local(output_pre)
        output_logit = self.convs[-1](output_pre)
        output_global = output_logit.view(output_logit.shape[0], -1)
        output = self.final(output_logit.view(output_logit.shape[0], -1))
        # print(output_logit.shape, output_local.shape, output_global.shape, output.shape, output.squeeze().shape)
        return output.squeeze(), output_global, output_local


class Discriminator_zz(nn.Module):
    def __init__(self, latent_size, dropout, output_size=1, do_spectral_norm=True):
        super(Discriminator_zz, self).__init__()
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
        intermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(intermediate_layer)
        return logit.squeeze(), intermediate_layer


class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_size

        self.output_bias = nn.Parameter(torch.zeros(3, 32, 32), requires_grad=True)
        self.deconv = nn.ModuleList()
        self.deconv.append(nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.latent_dim, 1024, 5, stride=2, bias=True, padding=1)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(1024, 512, 4, stride=1, bias=True)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(512, 256, 3, stride=2, bias=True)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(256, 128, 3, stride=2, bias=True)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(128, 64, 4, stride=1, bias=True)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ))
        self.deconv.append(nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(64, 3, 3, stride=1, bias=True)),
            nn.Tanh(),
        ))
        # self.init_weights()

    def forward(self, noise):
        x = noise
        for layer in self.deconv:
            x = layer(x)
            # print(x.shape)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


class Encoder(nn.Module):
    """ Encoder module of the CIFAR10 and SVHN datasets.
    It uses spectral normalization."""
    def __init__(self, latent_size, dropout=0.3, noise=False):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.latent_size = latent_size
        self.leaky_value = 0.2
        self.conv = nn.ModuleList()

        if noise:
            self.latent_size *= 2

        self.nonlinear = nn.LayerNorm(self.latent_size)
        # self.nonlinear = nn.Tanh()
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 5, stride=2, bias=True)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),

            SpectralNorm(nn.Conv2d(64, 128, 5, stride=1, bias=True)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 5, stride=1, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),

            SpectralNorm(nn.Conv2d(256, 512, 4, stride=1, bias=True)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(512, self.latent_size, 3, stride=1, bias=True)),
        ))
        # self.init_weights()

    def forward(self, x):
        output = x
        for layer in self.conv[:-1]:
            output = layer(output)
        output = self.conv[-1](output)
        output = self.nonlinear(output.view(output.shape[0], -1))
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        return output


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            SpectralNorm(nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=1):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)),
            nn.PReLU()
            )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(128))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU())

        self.deconv1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2)),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            SpectralNorm(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            SpectralNorm(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            )

        self.deconv2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            )

        # Final output layer
        self.conv_final = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, out_channels, 3, 1, padding=1)), nn.Tanh())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.deconv1(out)
        out1 = self.res_blocks(out)
        out = torch.add(out, out1)
        out = self.deconv2(out)
        out = self.conv_final(out)
        return out


#####################################################################################

class Discriminator_CD(nn.Module):
    """ Disciminator works on images of size (3, 64, 64).
    It uses spectral normalization. """
    def __init__(self, latent_size, dropout=0.3, output_size=1):
        super(Discriminator_CD, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2
        self.convs = nn.ModuleList()

        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 7, stride=1, bias=True, padding=1)),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),

            SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, bias=True)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 5, stride=1, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            # nn.Dropout2d(self.dropout),

            SpectralNorm(nn.Conv2d(256, 512, 5, stride=2, bias=True)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout), 
        ))
        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(512, 1024, 5, stride=1, bias=True)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(1024, 2048, 5, stride=2, bias=True)),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),
        ))

        self.local = SpectralNorm(nn.Conv2d(512, 2048, 1, 1))
        self.final = nn.Sequential(
            SpectralNorm(nn.Linear(2048, 2048, bias=True)),
            nn.ReLU(),
            SpectralNorm(nn.Linear(2048, self.output_size, bias=True)),
            )

    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
        output_local = self.local(output_pre)
        output_logit = self.convs[-1](output_pre)
        output_global = output_logit.view(output_logit.shape[0], -1)
        output = self.final(output_logit.view(output_logit.shape[0], -1))
        return output.squeeze(), output_global, output_local


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
            nn.LeakyReLU(self.leaky_value, inplace=True)
            )
        self.logit = nn.Sequential(
            SpectralNorm(nn.Linear(32, 1)),
            nn.LeakyReLU(self.leaky_value, inplace=True)
            )

    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        intermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(intermediate_layer)
        return logit.squeeze(), intermediate_layer


class Encoder_CD(nn.Module):
    """ Encoder module of the CIFAR10 and SVHN datasets.
    It uses spectral normalization."""
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
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 64, 7, stride=2, bias=True)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),

            SpectralNorm(nn.Conv2d(64, 128, 5, stride=1, bias=True)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True)
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 7, stride=2, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),

            SpectralNorm(nn.Conv2d(256, 512, 5, stride=1, bias=True)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(512, self.latent_size, 6, stride=1, bias=True)),
        ))
        # self.init_weights()

    def forward(self, x):
        output = x
        for layer in self.conv[:-1]:
            output = layer(output)
        output = self.conv[-1](output)
        output = self.nonlinear(output.view(output.shape[0], -1))
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        return output


class GeneratorResNet_CD(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=1):
        super(GeneratorResNet_CD, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)),
            nn.PReLU()
            )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(128))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU())

        self.deconv1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(512, 512, kernel_size=5, stride=2)),
            nn.BatchNorm2d(512),
            nn.PReLU(),

            SpectralNorm(nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2)),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            SpectralNorm(nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            )

        self.deconv2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            SpectralNorm(nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            )

        # Final output layer
        self.conv_final = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, out_channels, 3, 1, padding=1)), nn.Tanh())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.deconv1(out)
        out1 = self.res_blocks(out)
        out = torch.add(out, out1)
        out = self.deconv2(out)
        out = self.conv_final(out)
        return out


#####################################################################################
class Discriminator_MNIST(nn.Module):
    # images of size (1, 28, 28), FashionMNIST and MNIST
    def __init__(self, latent_size, dropout=0.3, output_size=1):
        super(Discriminator_MNIST, self).__init__()
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
            SpectralNorm(nn.Conv2d(64, 128, 3, stride=2, bias=True)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.convs.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),

            SpectralNorm(nn.Conv2d(256, 512, 3, stride=2, bias=True)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.local = SpectralNorm(nn.Conv2d(128, 512, 1, 1))
        self.final = nn.Sequential(
            SpectralNorm(nn.Linear(512, 2048, bias=True)),
            nn.ReLU(),
            SpectralNorm(nn.Linear(2048, self.output_size, bias=True))
            )

    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
        output_local = self.local(output_pre)
        output_logit = self.convs[-1](output_pre)
        output_global = output_logit.view(output_logit.shape[0], -1)
        output = self.final(output_logit.view(output_logit.shape[0], -1))
        return output.squeeze(), output_global, output_local


class Discriminator_zz_MNIST(nn.Module):
    def __init__(self, latent_size, dropout=0.3, output_size=1):
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
        intermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(intermediate_layer)
        return logit.squeeze(), intermediate_layer


class GeneratorResNet_MNIST(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=1):
        super(GeneratorResNet_MNIST, self).__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)),
            nn.PReLU(),
            )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(128))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU())

        self.deconv1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2)),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            SpectralNorm(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            )

        self.deconv2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            SpectralNorm(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            )

        # Final output layer
        self.conv_final = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, out_channels, 3, 1)), nn.Tanh())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.deconv1(out)
        out1 = self.res_blocks(out)
        out = torch.add(out, out1)
        out = self.deconv2(out)
        out = self.conv_final(out)
        return out


class Encoder_MNIST(nn.Module):
    def __init__(self, latent_size, dropout=0.3, noise=False):
        super(Encoder_MNIST, self).__init__()
        self.dropout = dropout
        self.latent_size = latent_size
        self.leaky_value = 0.1
        self.conv = nn.ModuleList()
        if noise:
            self.latent_size *= 2

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

            SpectralNorm(nn.Conv2d(256, 256, 3, stride=1, bias=True)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
        ))
        self.conv.append(nn.Sequential(
            SpectralNorm(nn.Conv2d(256, self.latent_size, 3, 1, bias=True))
        ))

    def forward(self, x):
        output = x
        for layer in self.conv[:-1]:
            output = layer(output)
        output = self.conv[-1](output)
        output = self.nonlinear(output.view(output.shape[0], -1))
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        return output
