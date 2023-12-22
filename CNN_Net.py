# 　VggNet
import torch
import torch.nn as nn
import torch.nn.functional as F

# CIFAR10输入图片尺寸（3*32*32）crop-->(3*28*28)

class VggBase(nn.Module):
	def __init__(self):
		super(VggBase, self).__init__()

		# batchsize*3*28*28
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)
		self.max_poooling1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

		# batchsize*64*14*14
		self.conv2_1 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)
		self.conv2_2 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)
		self.max_poooling2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

		# batchsize*128*7*7
		self.conv3_1 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(256),
			nn.ReLU()
		)
		self.conv3_2 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(256),
			nn.ReLU()
		)
		self.max_poooling3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)

		# batchsize*256*4*4
		self.conv4_1 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(512),
			nn.ReLU()
		)
		self.conv4_2 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(512),
			nn.ReLU()
		)
		self.max_poooling4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

		# batchsize*512*2*2
		# batchsize*512*2(reshape)-->batchsize*512*4
		self.fc = nn.Linear(512 * 4, 10)

	def forward(self, x):
		batchsize = x.size(0)
		out = self.max_poooling1(self.conv1(x))
		out = self.max_poooling2(self.conv2_2(self.conv2_1(out)))
		out = self.max_poooling3(self.conv3_2(self.conv3_1(out)))
		out = self.max_poooling4(self.conv4_2(self.conv4_1(out)))
		out = out.view(batchsize, -1) # batchsize*c*h*w-->batchsize*(c*h*w)
		out = self.fc(out)
		out = F.log_softmax(out, dim=1)
		return out

def VggNet():
	return VggBase()






