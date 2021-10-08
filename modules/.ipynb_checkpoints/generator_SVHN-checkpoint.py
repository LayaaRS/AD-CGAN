import torch
import torch.nn as nn

class Generator(nn.Module):  
	def __init__(self, latent_size):
		super(Generator, self).__init__()
		self.latent_dim = latent_size
		self.leaky_value = 0.2
		
		self.output_bias = nn.Parameter(torch.zeros(3, 32, 32), requires_grad=True)
		self.deconv = nn.ModuleList()
		self.deconv.append(nn.Sequential(
			nn.ConvTranspose2d(self.latent_dim, 1024, 4, stride=2, bias=True, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
		))
		self.deconv.append(nn.Sequential(    
			nn.ConvTranspose2d(1024, 512, 4, stride=2, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
		))
		self.deconv.append(nn.Sequential(    
			nn.ConvTranspose2d(512, 512, 4, stride=2, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
		))
		self.deconv.append(nn.Sequential(    
			nn.ConvTranspose2d(512, 512, 5, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
		))
		self.deconv.append(nn.Sequential(    
			nn.ConvTranspose2d(512, 512, 5, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
		))
		self.deconv.append(nn.Sequential(    
			nn.ConvTranspose2d(512, 256, 3, stride=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
		))
		self.deconv.append(nn.Sequential(    
			nn.ConvTranspose2d(256, 256, 3, stride=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
		))
		self.deconv.append(nn.Sequential(    
			nn.ConvTranspose2d(256, 128, 3, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
		))
		self.deconv.append(nn.Sequential(    
			nn.ConvTranspose2d(128, 64, 3, stride=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
		))
		self.deconv.append(nn.Sequential(    
			nn.ConvTranspose2d(64, 3, 3, stride=1, bias=True),
            nn.Tanh()
		))
		
		self.init_weights()

	def forward(self, noise):
		x = noise
		for layer in self.deconv:
			x = layer(x)
		# output = torch.sigmoid(x)
		return x

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