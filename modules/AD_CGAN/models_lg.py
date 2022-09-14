import torch
import torch.nn as nn
from .spectral_normalization import SpectralNorm


# 32 * 32 * 3 images
class Discriminator(nn.Module):
    """Disciminator works on images of size (3, 32, 32).
    It uses spectral normalization."""

    def __init__(self, latent_size, dropout=0.3, output_size=1):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2
        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                SpectralNorm(
                    nn.Conv2d(3, 64, 5, stride=1, bias=True, padding=1)),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
                SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(128, 256, 5, stride=1, bias=True)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                # nn.Dropout2d(self.dropout),
                SpectralNorm(nn.Conv2d(256, 512, 5, stride=1, bias=True)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
            )
        )
        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(512, 1024, 3, stride=1, bias=True)),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                SpectralNorm(nn.Conv2d(1024, 2048, 3, stride=2, bias=True)),
                nn.BatchNorm2d(2048),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
            )
        )

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
        # output_logit = self.convs[-1](output_pre)

        output_global = self.convs[-1](output_pre)
        output_global = torch.sum(output_global, dim=(2, 3))

        # output_global = output_logit.view(output_logit.shape[0], -1)
        # output = self.final(output_logit.view(output_logit.shape[0], -1))

        output = self.final(output_global)

        # print('our local global features')
        # print(output.squeeze().shape, output_global.shape, output_local.shape)
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
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )
        self.logit = nn.Sequential(SpectralNorm(
            nn.Linear(32, 1)), nn.LeakyReLU(self.leaky_value, inplace=True))

    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        intermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(intermediate_layer)
        return logit.squeeze(), intermediate_layer


class Encoder(nn.Module):
    """Encoder module of the CIFAR10 and SVHN datasets.
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
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(3, 64, 5, stride=2, bias=True)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
                SpectralNorm(nn.Conv2d(64, 128, 5, stride=1, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(128, 256, 5, stride=1, bias=True)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
                SpectralNorm(nn.Conv2d(256, 512, 4, stride=1, bias=True)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
            )
        )
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(512, self.latent_size,
                             3, stride=1, bias=True)),
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


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_features, in_features,
                         kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            SpectralNorm(nn.Conv2d(in_features, in_features,
                         kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=1):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, 128,
                         kernel_size=3, stride=1, padding=1)),
            nn.PReLU(),
            # SpectralNorm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            # nn.BatchNorm2d(128),
            # nn.PReLU()
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(128))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1,
                      padding=1), nn.BatchNorm2d(512), nn.PReLU()
        )

        self.deconv1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(
                512, 512, kernel_size=4, stride=2)),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            SpectralNorm(nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            SpectralNorm(nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.deconv2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        # Final output layer
        self.conv_final = nn.Sequential(SpectralNorm(
            nn.Conv2d(64, out_channels, 3, 1, padding=1)), nn.Tanh())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.deconv1(out)
        # print(out.shape)
        out1 = self.res_blocks(out)
        # print(out1.shape)
        out = torch.add(out, out1)
        # print(out1.shape)
        out = self.deconv2(out)
        # print(out.shape)
        out = self.conv_final(out)
        # print(out.shape)
        return out


#####################################################################################
# 28 * 28 * 1 images
class Discriminator_MNIST(nn.Module):
    def __init__(self, latent_size, dropout=0.3, output_size=1, nrkhs=1024):
        super(Discriminator_MNIST, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.1
        self.nrkhs = nrkhs

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(1, 64, 3, stride=2, bias=True)),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(128, 128, 3, stride=2, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                # SpectralNorm(nn.Conv2d(256, 512, 3, stride=2, bias=True)),
                # nn.BatchNorm2d(512),
                # nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.local = SpectralNorm(nn.Conv2d(128, 128, 1, 1))
        self.final = nn.Sequential(
            SpectralNorm(nn.Linear(128, 256, bias=True)),
            nn.ReLU(),
            SpectralNorm(nn.Linear(256, self.output_size, bias=True))
        )

    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
            # print(output_pre.shape)
        output_local = self.local(output_pre)
        # print('InfoMAX local global features')
        # print('local', output_local.shape)
        output_global = self.convs[-1](output_local)
        # print("global", output_global.shape)
        output_global = torch.sum(output_global, dim=(2, 3))
        # print('global', output_global.shape)
        output = self.final(output_global)
        # print('output', output.squeeze().shape)
        return output.squeeze(), output_global, output_local


class Discriminator_MNIST_infomax(nn.Module):
    def __init__(self, latent_size, dropout=0.3, output_size=1, nrkhs=1024, ndf=128):
        super(Discriminator_MNIST_infomax, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.1
        self.nrkhs = nrkhs

        self.ndf = ndf
        self.nrkhs = nrkhs

        self.activation = nn.ReLU(True)

        self.local_nrkhs_a = SpectralNorm(
            nn.Conv2d(self.ndf, self.ndf, 1, 1, 0))
        self.local_nrkhs_b = SpectralNorm(
            nn.Conv2d(self.ndf, self.nrkhs, 1, 1, 0))
        self.local_nrkhs_sc = SpectralNorm(
            nn.Conv2d(self.ndf, self.nrkhs, 1, 1, 0))

        self.global_nrkhs_a = SpectralNorm(nn.Linear(self.ndf, self.ndf))
        self.global_nrkhs_b = SpectralNorm(nn.Linear(self.ndf, self.nrkhs))
        self.global_nrkhs_sc = SpectralNorm(nn.Linear(self.ndf, self.nrkhs))

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(1, 64, 3, stride=2, bias=True)),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(128, 128, 3, stride=2, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                SpectralNorm(nn.Conv2d(128, 128, 3, stride=1, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                SpectralNorm(nn.Conv2d(128, 128, 3, stride=1, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.local = SpectralNorm(nn.Conv2d(128, 128, 1, 1))
        self.linear = SpectralNorm(nn.Linear(self.ndf, 1))
        self.final = nn.Sequential(
            SpectralNorm(nn.Linear(128, 1, bias=True)),
            # nn.ReLU(),
            # SpectralNorm(nn.Linear(self.nrkhs, self.output_size, bias=True))
        )

    def _project_local(self, local_feat):
        r"""
        Helper function for projecting local features to RKHS.
        """
        local_feat_sc = self.local_nrkhs_sc(local_feat)

        local_feat = self.local_nrkhs_a(local_feat)
        local_feat = self.activation(local_feat)
        local_feat = self.local_nrkhs_b(local_feat)
        local_feat += local_feat_sc

        return local_feat

    def _project_global(self, global_feat):
        r"""
        Helper function for projecting global features to RKHS.
        """
        global_feat_sc = self.global_nrkhs_sc(global_feat)

        global_feat = self.global_nrkhs_a(global_feat)
        global_feat = self.activation(global_feat)
        global_feat = self.global_nrkhs_b(global_feat)
        global_feat += global_feat_sc

        return global_feat

    def project_features(self, local_feat, global_feat):
        r"""
        Projects local and global features.
        """
        local_feat = self._project_local(
            local_feat)  # (N, C, H, W) --> (N, nrkhs, H, W)
        global_feat = self._project_global(
            global_feat)  # (N, C) --> (N, nrkhs)

        return local_feat, global_feat

    # Project the features
    def forward(self, x):
        output_pre = x
        for layer in self.convs[:-1]:
            output_pre = layer(output_pre)
            # print(output_pre.shape)
        output_local = self.local(output_pre)
        # print('InfoMAX local global features')
        output_global = self.convs[-1](output_local)
        # output_global = self.activation(output_global)
        output_global = torch.sum(output_global, dim=(2, 3))
        output = self.final(output_global)
        output_local, output_global = self.project_features(
            local_feat=output_local, global_feat=output_global.view(
                output_global.shape[0], -1)
        )
        # print('output', output.squeeze().shape)
        # print(output.squeeze().shape, output_global.shape, output_local.shape)
        return output.squeeze(), output_global, output_local


class local_global(nn.Module):
    def __init__(self, nrkhs=1024, ndf=128):
        super(local_global, self).__init__()
        self.ndf = ndf
        self.nrkhs = nrkhs

        self.activation = nn.ReLU(True)

        self.local_nrkhs_a = SpectralNorm(
            nn.Conv2d(self.ndf, self.ndf, 1, 1, 0))
        self.local_nrkhs_b = SpectralNorm(
            nn.Conv2d(self.ndf, self.nrkhs, 1, 1, 0))
        self.local_nrkhs_sc = SpectralNorm(
            nn.Conv2d(self.ndf, self.nrkhs, 1, 1, 0))

        self.global_nrkhs_a = SpectralNorm(nn.Linear(self.ndf, self.ndf))
        self.global_nrkhs_b = SpectralNorm(nn.Linear(self.ndf, self.nrkhs))
        self.global_nrkhs_sc = SpectralNorm(nn.Linear(self.ndf, self.nrkhs))

    def _project_local(self, local_feat):
        r"""
        Helper function for projecting local features to RKHS.
        """
        local_feat_sc = self.local_nrkhs_sc(local_feat)

        local_feat = self.local_nrkhs_a(local_feat)
        local_feat = self.activation(local_feat)
        local_feat = self.local_nrkhs_b(local_feat)
        local_feat += local_feat_sc

        return local_feat

    def _project_global(self, global_feat):
        r"""
        Helper function for projecting global features to RKHS.
        """
        global_feat_sc = self.global_nrkhs_sc(global_feat)

        global_feat = self.global_nrkhs_a(global_feat)
        global_feat = self.activation(global_feat)
        global_feat = self.global_nrkhs_b(global_feat)
        global_feat += global_feat_sc

        return global_feat

    def project_features(self, local_feat, global_feat):
        r"""
        Projects local and global features.
        """
        local_feat = self._project_local(
            local_feat)  # (N, C, H, W) --> (N, nrkhs, H, W)
        global_feat = self._project_global(
            global_feat)  # (N, C) --> (N, nrkhs)

        return local_feat, global_feat

    # Project the features
    def forward(self, local_feat_real, global_feat_real):
        local_feat_real, global_feat_real = self.project_features(
            local_feat=local_feat_real, global_feat=global_feat_real
        )
        # print(local_feat_real.shape, global_feat_real.shape)
        return local_feat_real, global_feat_real


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
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )
        self.logit = nn.Sequential(SpectralNorm(
            nn.Linear(32, 1)), nn.LeakyReLU(self.leaky_value, inplace=True))

    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        # print(z.shape)
        intermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(intermediate_layer)
        # print(intermediate_layer.shape, logit.squeeze().shape)
        return logit.squeeze(), intermediate_layer


class Generator_MNIST(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=1):
        super(Generator_MNIST, self).__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, 128,
                         kernel_size=3, stride=1, padding=1)),
            nn.PReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(128))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1,
                      padding=1), nn.BatchNorm2d(512), nn.PReLU()
        )

        self.deconv1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2)),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            SpectralNorm(nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.deconv2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            SpectralNorm(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        # Final output layer
        self.conv_final = nn.Sequential(SpectralNorm(
            nn.Conv2d(64, out_channels, 3, 1)), nn.Tanh())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.deconv1(out)
        # print(out.shape)
        out1 = self.res_blocks(out)
        # print(out1.shape)
        out = torch.add(out, out1)
        # print(out1.shape)
        out = self.deconv2(out)
        # print(out.shape)
        out = self.conv_final(out)
        # print(out.shape)
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
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(1, 64, 3, stride=1, bias=True)),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(64, 128, 3, stride=2, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(128, 256, 3, stride=2, bias=True)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                SpectralNorm(nn.Conv2d(256, 256, 3, stride=1, bias=True)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.conv.append(nn.Sequential(SpectralNorm(
            nn.Conv2d(256, self.latent_size, 3, 1, bias=True))))

    def forward(self, x):
        output = x
        for layer in self.conv[:-1]:
            output = layer(output)
        output = self.conv[-1](output)
        output = self.nonlinear(output.view(output.shape[0], -1))
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        return output


class Discriminator_CD(nn.Module):
    """Disciminator works on images of size (3, 64, 64).
    It uses spectral normalization."""

    def __init__(self, latent_size, dropout=0.3, output_size=1):
        super(Discriminator_CD, self).__init__()
        self.latent_size = latent_size
        self.dropout = dropout
        self.output_size = output_size
        self.leaky_value = 0.2
        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                SpectralNorm(
                    nn.Conv2d(3, 64, 7, stride=1, bias=True, padding=1)),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
                SpectralNorm(nn.Conv2d(64, 128, 5, stride=2, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(128, 256, 5, stride=1, bias=True)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                # nn.Dropout2d(self.dropout),
                SpectralNorm(nn.Conv2d(256, 512, 5, stride=2, bias=True)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
            )
        )
        self.convs.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(512, 1024, 5, stride=1, bias=True)),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                SpectralNorm(nn.Conv2d(1024, 2048, 5, stride=2, bias=True)),
                nn.BatchNorm2d(2048),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
            )
        )

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

        output_global = self.convs[-1](output_pre)
        output_global = torch.sum(output_global, dim=(2, 3))
        output = self.final(output_global)

        # print(output.squeeze().shape, output_global.shape, output_local.shape)

        # output_logit = self.convs[-1](output_pre)
        # output_global = output_logit.view(output_logit.shape[0], -1)
        # output = self.final(output_logit.view(output_logit.shape[0], -1))
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
            nn.LeakyReLU(self.leaky_value, inplace=True),
        )
        self.logit = nn.Sequential(SpectralNorm(
            nn.Linear(32, 1)), nn.LeakyReLU(self.leaky_value, inplace=True))

    def forward(self, z, z_prime):
        z = torch.cat((z, z_prime), 0)
        intermediate_layer = self.joint_zz(z.view(z.shape[0], -1))
        logit = self.logit(intermediate_layer)
        return logit.squeeze(), intermediate_layer


class Encoder_CD(nn.Module):
    """Encoder module of the CIFAR10 and SVHN datasets.
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
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(3, 64, 7, stride=2, bias=True)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
                SpectralNorm(nn.Conv2d(64, 128, 5, stride=1, bias=True)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(self.leaky_value, inplace=True),
            )
        )
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(128, 256, 7, stride=2, bias=True)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
                SpectralNorm(nn.Conv2d(256, 512, 5, stride=1, bias=True)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(self.leaky_value, inplace=True),
                nn.Dropout2d(self.dropout),
            )
        )
        self.conv.append(
            nn.Sequential(
                SpectralNorm(nn.Conv2d(512, self.latent_size,
                             6, stride=1, bias=True)),
            )
        )
        # self.init_weights()

    def forward(self, x):
        output = x
        for layer in self.conv[:-1]:
            output = layer(output)
        output = self.conv[-1](output)
        output = self.nonlinear(output.view(output.shape[0], -1))
        output = output.view((output.shape[0], self.latent_size, 1, 1))
        return output


class Generator_CD(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=1):
        super(Generator_CD, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, 128, kernel_size=3,
                         stride=1, padding=1)), nn.PReLU()
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(128))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1,
                      padding=1), nn.BatchNorm2d(512), nn.PReLU()
        )

        self.deconv1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(
                512, 512, kernel_size=5, stride=2)),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            SpectralNorm(nn.ConvTranspose2d(
                512, 256, kernel_size=5, stride=2)),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            SpectralNorm(nn.ConvTranspose2d(
                256, 128, kernel_size=5, stride=2)),
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
        self.conv_final = nn.Sequential(SpectralNorm(
            nn.Conv2d(64, out_channels, 3, 1, padding=1)), nn.Tanh())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.deconv1(out)
        out1 = self.res_blocks(out)
        out = torch.add(out, out1)
        out = self.deconv2(out)
        out = self.conv_final(out)
        return out
