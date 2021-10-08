import torch
import torch.nn as nn

class Discriminator(nn.Module):
	""" the disciminator works on images of size (3, 32, 32) """
	def __init__(self, latent_size, dropout=0.3, output_size=2):
		super(Discriminator, self).__init__()
		self.latent_size = latent_size
		self.dropout = dropout
		self.output_size = output_size
		self.leaky_value = 0.2

		self.convs = nn.ModuleList()

		self.convs.append(nn.Sequential(
			nn.Conv2d(3, 64, 5, stride=2, bias=True),
			# nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),
		))
		self.convs.append(nn.Sequential(
			nn.Conv2d(64, 128, 5, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.leaky_value, inplace=True)
		))
		self.convs.append(nn.Sequential(
			nn.Conv2d(128, 256, 5, stride=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.leaky_value, inplace=True),
            nn.Dropout2d(self.dropout),
		))
		self.convs.append(nn.Sequential(
			nn.Conv2d(256, 512, 4, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.leaky_value , inplace=True),
            nn.Dropout2d(self.dropout),
		))
		self.final = nn.Sequential(
			# nn.Linear(512, self.output_size, bias=True)
			nn.Conv2d(512, self.latent_size, 3, stride=1, bias=True),
			)
		
		self.init_weights()
		
	def forward(self, x):
		output_pre = x
		for layer in self.convs[:-1]:
			output_pre = layer(output_pre)
			# print(output_pre.shape)
		output = self.convs[-1](output_pre)
		output = self.final(output)
		# print(output.shape)
		# output = self.final(output.view(output.shape[0],-1))
		output = output.view((output.shape[0], self.latent_size, 1, 1))
		return output.squeeze(), output_pre

	def init_weights(self):
		for m in self.modules():
			if isinstance(m , nn.Conv2d):
				torch.nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
			elif isinstance(m, nn.ConvTranspose2d):
				torch.nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
			elif isinstance(m, nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)