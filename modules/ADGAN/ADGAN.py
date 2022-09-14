import torch.nn as nn

#  All images rescaled to (3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self, latent_size, input_size):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size
        # nc = 1 if self.input_size == 32 else 3

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                nn.Conv2d(self.input_size, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        self.final = nn.Sequential(
            nn.Linear(256 * 4 * 4, self.latent_size), nn.Tanh())

    def forward(self, x):
        for layers in self.convs:
            output = layers(x)
        output = self.final(output.view(output.shape[0], -1))
        return output


class Generator(nn.Module):
    def __init__(self, latent_size, input_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size

        self.deconvs = nn.ModuleList()

        self.linear = nn.Sequential(
            nn.Linear(self.latent_size, 64 * 4 * 4 *
                      4), nn.BatchNorm1d(64 * 4 * 4 * 4), nn.ReLU()
        )

        self.deconvs.append(nn.Sequential(nn.ConvTranspose2d(
            64 * 4, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU()))

        self.deconvs.append(nn.Sequential(nn.ConvTranspose2d(
            128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU()))

        self.deconvs.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, self.input_size, 4, 2, 1),
                nn.Tanh(),
            )
        )

    def forward(self, x):
        output = self.linear(x.view(x.shape[0], -1))
        output = output.reshape(x.shape[0], 64 * 4, 4, 4)
        for layers in self.deconvs:
            output = layers(output)
        return output

#  images of the size (3, 64, 64)


class Discriminator_CD(nn.Module):
    def __init__(self, latent_size, input_size):
        super(Discriminator_CD, self).__init__()
        self.latent_size = latent_size
        self.input_size = input_size

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                nn.Conv2d(self.input_size, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(128, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            ))

        self.final = nn.Sequential(
            nn.Linear(256*4*4, self.latent_size),
            nn.Tanh()
        )

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
            nn.Linear(self.latent_size, 64*4*4*4),
            nn.BatchNorm1d(64*4*4*4),
            nn.ReLU()
        )

        self.deconvs.append(nn.Sequential(
            nn.ConvTranspose2d(64*4, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ))

        self.deconvs.append(nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ))

        self.deconvs.append(nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ))

        self.deconvs.append(nn.Sequential(
            nn.ConvTranspose2d(64, self.input_size, 4, 2, 1),
            nn.Tanh(),
        ))

    def forward(self, x):
        output = self.Linear(x.view(x.shape[0], -1))
        output = output.reshape(x.shape[0], 64*4, 4, 4)
        for layers in self.deconvs:
            output = layers(output)
        return output


# --------------------------------------------------------------------------------------------------------------------
# -------- In case of testing contrastive GAN for ADGAN uncomment the below and comment disciriminators above --------
# --------------------------------------------------------------------------------------------------------------------

# class Discriminator_CD(nn.Module):
#     def __init__(self, latent_size, input_size):
#         super(Discriminator_CD, self).__init__()
#         self.latent_size = latent_size
#         self.input_size = input_size

#         self.convs = nn.ModuleList()

#         self.convs.append(nn.Sequential(
#             nn.Conv2d(self.input_size, 64, 3, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 128, 3, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 256, 3, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()))

#         self.convs.append(nn.Sequential(
#             nn.Conv2d(256, 256, 3, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()))

#         self.local = nn.Conv2d(256, 4096, 1, 1)
#         self.final = nn.Sequential(
#             nn.Linear(256*4*4, self.latent_size),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         output_pre = x
#         for layers in self.convs[:-1]:
#             output_pre = layers(output_pre)
#         output_local = self.local(output_pre)
#         output_logit = self.convs[-1](output_pre)
#         output_global = output_logit.view(output_logit.shape[0], -1)
#         output = self.final(output_logit.view(output_logit.shape[0], -1))
#         return output.squeeze(), output_global, output_local


# class Discriminator(nn.Module):
#     def __init__(self, latent_size, input_size):
#         super(Discriminator, self).__init__()
#         self.latent_size = latent_size
#         self.input_size = input_size

#         self.convs = nn.ModuleList()

#         self.convs.append(nn.Sequential(
#             nn.Conv2d(self.input_size, 64, 3, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 128, 3, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()))

#         self.convs.append(nn.Sequential(
#             nn.Conv2d(128, 256, 3, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()))

#         self.local = nn.Conv2d(128, 4096, 1, 1)
#         self.final = nn.Sequential(
#             nn.Linear(256*4*4, self.latent_size),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         output_pre = x
#         for layers in self.convs[:-1]:
#             output_pre = layers(output_pre)
#         output_local = self.local(output_pre)
#         output_logit = self.convs[-1](output_pre)
#         output_global = output_logit.view(output_logit.shape[0], -1)
#         output = self.final(output_logit.view(output_logit.shape[0], -1))
#         return output.squeeze(), output_global, output_local
