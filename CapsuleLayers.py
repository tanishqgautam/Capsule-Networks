import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def squash(inputs, axis=-1):
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class PrimaryCapsule(nn.Module):
	def __init__(self, num_maps=32, num_dims=8):		
		super(PrimaryCapsule, self).__init__()
		self.num_maps = num_maps
		self.num_caps = 6 * 6 * self.num_maps
		self.num_dims = num_dims
		self.conv1 = nn.Conv2d(256, self.num_maps * self.num_dims, kernel_size=9, stride=2, padding=0)

	def forward(self, x):
		# 20, 20, 256
		out = self.conv1(x)
		# 6, 6, 256
		out = out.view(-1, self.num_caps, self.num_dims)
		out = squash(out)
		return out

class DenseCapsule(nn.Module):
	def __init__(self, num_caps_in, num_caps_out, num_dims_in, num_dims_out, routings=3):		
		super(DenseCapsule, self).__init__()
		self.weight = nn.Parameter(.01 * torch.randn(num_caps_out, num_caps_in, num_dims_out, num_dims_in))
		self.routings = routings
		self.num_caps_in = num_caps_in
		self.num_caps_out = num_caps_out

	def forward(self, x):
		x = x[:, None, :, :, None] #expands dims
		x_hat = torch.squeeze(torch.matmul(self.weight, x), dim=-1)

		x_hat_detached = x_hat.detach()

		b = Variable(torch.zeros(x.shape[0], self.num_caps_out, self.num_caps_in).cuda())

		assert self.routings > 0
		for i in range(self.routings):
			c = F.softmax(b, dim=1)
			if i == self.routings - 1:
				out = squash(torch.sum(c[:,:,:, None] * x_hat, dim=-2, keepdim=True))
			else: #no gradeinets here
				out = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
				b = b + torch.sum(out * x_hat_detached, dim =-1)
		return torch.squeeze(out, dim=-2)
