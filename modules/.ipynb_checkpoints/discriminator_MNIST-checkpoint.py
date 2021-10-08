import torch
import torch.nn as nn

# class Discriminator(nn.Module):

# 	def __init__(self, latent_size, dropout=0.2, output_size=2):
# 		super(Discriminator, self).__init__()
# 		self.latent_size = latent_size
# 		self.dropout = dropout
# 		self.output_size = output_size
# 		self.leaky_value = 0.1

# 		self.convs = nn.ModuleList()

# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(1, 64, 5, stride=2, bias=True),
# 			nn.BatchNorm2d(64),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(64, 128, 3, stride=1, bias=True),
# 			nn.BatchNorm2d(128),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(128, 256, 3, stride=1, bias=True),
# 			nn.BatchNorm2d(256),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(256, 512, 3, stride=1, bias=True),
# 			nn.BatchNorm2d(512),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(512, 512, 3, stride=1, bias=True),
# 			nn.BatchNorm2d(512),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(512, 256, 3, stride=1, bias=True),
# 			nn.BatchNorm2d(256),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 		))

# 		# self.final = nn.Conv2d(512, self.output_size, 1, stride=1, bias=True)
# 		self.final = nn.Linear(1024, self.output_size, bias=True)
		
# 		self.init_weights()
		
# 	def forward(self, x):
# 		output_pre = x
# 		for layer in self.convs[:-1]:
# 			output_pre = layer(output_pre)

# 		output = self.convs[-1](output_pre)
# 		output = self.final(output.view(output.shape[0],-1))
# 		# output = torch.sigmoid(output)		
# 		return output.squeeze(), output_pre

# 	def init_weights(self):
# 		for m in self.modules():
# 			if isinstance(m , nn.Conv2d):
# 				torch.nn.init.xavier_uniform_(m.weight)
# 				m.bias.data.fill_(0.01)
# 			elif isinstance(m, nn.ConvTranspose2d):
# 				torch.nn.init.xavier_uniform_(m.weight)
# 				m.bias.data.fill_(0.01)
# 			elif isinstance(m, nn.Linear):
# 				torch.nn.init.xavier_uniform_(m.weight)
# 				m.bias.data.fill_(0.01)

############################ the final one

class Discriminator(nn.Module):

	def __init__(self, latent_size, dropout=0.2, output_size=1):
		super(Discriminator, self).__init__()
		self.latent_size = latent_size
		self.dropout = dropout
		self.output_size = output_size
		self.leaky_value = 0.1

		self.convs = nn.ModuleList()

		self.convs.append(nn.Sequential(
			nn.Conv2d(1, 64, 3, stride=2, bias=True),
			nn.LeakyReLU(self.leaky_value, inplace=True),
		))
		self.convs.append(nn.Sequential(
			nn.Conv2d(64, 128, 3, stride=1, bias=True),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(self.leaky_value, inplace=True),

			nn.Conv2d(128, 256, 3, stride=2, bias=True),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(self.leaky_value, inplace=True),
		))

		self.final1 = nn.Linear(6400, 1024, bias=True)
		self.final2 = nn.Linear(1024, self.output_size, bias=True)
		
		self.init_weights()
		
	def forward(self, x):
		output_pre = x
		for layer in self.convs[:-1]:
			output_pre = layer(output_pre)
		output = self.convs[-1](output_pre)
		output = self.final1(output.view(output.shape[0],-1))
		output = self.final2(output)
		# output = torch.sigmoid(output)		
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



# class DiscriminatorBigan(nn.Module):

# 	def __init__(self, latent_size, dropout=0.2, output_size=2):
# 		super(DiscriminatorBigan, self).__init__()
# 		self.latent_size = latent_size
# 		self.dropout = dropout
# 		self.output_size = output_size
# 		self.leaky_value = 0.1

# 		self.convs = nn.ModuleList()
# 		# input size = 3 * 256 * 256
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(3, 64, 5, stride=2, bias=True),
# 			nn.BatchNorm2d(64),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 			# output size 64 * 126 * 126
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(64, 128, 5, stride=2, bias=True),
# 			nn.BatchNorm2d(128),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 			# output size 128 * 61 * 61
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(128, 256, 3, stride=1, bias=True),
# 			nn.BatchNorm2d(256),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 			nn.MaxPool2d(2, stride=2),
# 			# output size 256 * 29 * 29
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(256, 512, 3, stride=1, bias=True),
# 			nn.BatchNorm2d(512),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 			# output size 512 * 27 * 27
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(512, 512, 3, stride=1, bias=True),
# 			nn.BatchNorm2d(512),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 			# output size 512 * 25 * 25
# 		))
# 		self.convs.append(nn.Sequential(
# 			nn.Conv2d(512, 256, 3, stride=1, bias=True),
# 			nn.BatchNorm2d(256),
# 			nn.LeakyReLU(self.leaky_value, inplace=True),
# 			nn.Dropout2d(p=self.dropout),
# 			nn.MaxPool2d(2, stride=2),
# 			# output size 256 * 11 * 11
# 		))

# 		self.infer_joint = nn.Sequential(
# 			nn.Linear(30976+self.latent_size,1024),
# 			nn.LeakyReLU(self.leaky_value),
# 			nn.Linear(1024,512),
# 			nn.LeakyReLU(self.leaky_value),
# 			)

# 		self.final = nn.Linear(512, self.output_size, bias=True)
		
# 		self.init_weights()
		
# 	def forward(self, x, z):
		
# 		x_features = x

# 		for layer in self.convs[:-1]:
# 			x_features = layer(x_features)
			

# 		x_output = self.convs[-1](x_features)

# 		jointed = torch.cat((x_output.view(x_output.shape[0],-1),z.view(z.shape[0],-1)),1)
# 		jointed = self.infer_joint(jointed)
# 		output = self.final(jointed)
# 		# output = torch.sigmoid(output)
		
# 		return output.squeeze(), x_features

# 	def init_weights(self):
# 		for m in self.modules():
# 			if isinstance(m , nn.Conv2d):
# 				torch.nn.init.xavier_uniform_(m.weight)
# 				m.bias.data.fill_(0.01)
# 			elif isinstance(m, nn.ConvTranspose2d):
# 				torch.nn.init.xavier_uniform_(m.weight)
# 				m.bias.data.fill_(0.01)
# 			elif isinstance(m, nn.Linear):
# 				torch.nn.init.xavier_uniform_(m.weight)
# 				m.bias.data.fill_(0.01)