# 　VggNet
import torch
import torch.nn as nn
import torch.nn.functional as F


# CIFAR10输入图片尺寸（3*32*32）crop-->(3*28*28)

# Vgg部分
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
		x = self.max_poooling1(self.conv1(x))
		x = self.max_poooling2(self.conv2_2(self.conv2_1(x)))
		x = self.max_poooling3(self.conv3_2(self.conv3_1(x)))
		x = self.max_poooling4(self.conv4_2(self.conv4_1(x)))
		x = x.view(batchsize, -1)  # batchsize*c*h*w-->batchsize*(c*h*w)
		x = self.fc(x)
		x = F.log_softmax(x, dim = 1)
		return x

###########################################################################################
# Resnet部分
class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, stride=1):
		super(ResBlock, self).__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(in_channel, out_channel,
			          kernel_size = 3, stride = stride, padding = 1),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(),
			nn.Conv2d(out_channel, out_channel,
			          kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(out_channel)
		)
		self.shortcut = nn.Sequential()
		if in_channel != out_channel or stride > 1:  # 通过卷积操作进行下采样
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channel, out_channel,
				          kernel_size = 3, stride = stride, padding = 1),
				nn.BatchNorm2d(out_channel),
			)

	def forward(self, x):
		out1 = self.layer(x)
		out2 = self.shortcut(x)
		out = out1 + out2
		out = F.relu(out)
		return out
class Resnet(nn.Module):
	def make_layer(self, block, out_channel, stride, num_block):
		layers_list = [] #通过列表存放相应的层
		for i in range(num_block):
			if i == 0:
				in_stride = stride
			else:
				in_stride = 1
			layers_list.append(block(self.in_channel,
			                         out_channel,
			                         in_stride))
			self.in_channel = out_channel

		return nn.Sequential(*layers_list)

	def __init__(self, ResBlock):
		super(Resnet, self).__init__()
		self.in_channel = 32
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
			nn.BatchNorm2d(32),
			nn.ReLU()
		)

		# self.layer1 = ResBlock(in_channel =32 ,
		#                        out_channel = 64,
		#                        stride = 2)
		self.layer1 = \
			self.make_layer(ResBlock, 64, 2, 2)
		self.layer2 = \
			self.make_layer(ResBlock, 128, 2, 2)
		self.layer3 = \
			self.make_layer(ResBlock, 256, 2, 2)
		self.layer4 = \
			self.make_layer(ResBlock, 512, 2, 2)
		self.fc = nn.Linear(512, 10)

	def forward(self, x):
		out = self.conv1(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 2) # 将输出结果转换到1*1feature_map上
		out = out.view(out.size(0), -1) # 将特征图的维度拉成一维
		out = self.fc(out)
		return out

# MobileNet部分
class MobileNet(nn.Module):
	def con_dw(self, in_channel, out_channel, stride):
		return nn.Sequential(
			nn.Conv2d(in_channel, in_channel,
			          kernel_size = 3, stride = stride, padding = 1,
			          groups = in_channel, bias = False),
			nn.BatchNorm2d(in_channel),
			nn.ReLU(),

			nn.Conv2d(in_channel, out_channel,
			          kernel_size = 1, stride = 1, padding = 0,
			          bias = False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU()
		)

	def __init__(self):
		super(MobileNet, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1,1),
			nn.BatchNorm2d(32),
			nn.ReLU())
		self.con_dw2 = self.con_dw(32, 32, 1)
		self.con_dw3 = self.con_dw(32, 64, 2)
		self.con_dw4 = self.con_dw(64, 64, 1)
		self.con_dw5 = self.con_dw(64, 128, 2)
		self.con_dw6 = self.con_dw(128, 128, 1)
		self.con_dw7 = self.con_dw(128, 256, 2)
		self.con_dw8 = self.con_dw(256, 256, 1)
		self.con_dw9 = self.con_dw(256, 512, 2)
		self.fc = nn.Linear(512, 10)

	def forward(self, x):
		x = self.conv1(x)
		x= self.con_dw9(self.con_dw8(self.con_dw7(self.con_dw6(self.con_dw5(self.con_dw4(self.con_dw3(self.con_dw2(x))))))))
		x = F.avg_pool2d(x, 2)
		x = x.view(-1, 512)
		out = self.fc(x)
		return out

# InceptionNet
def ConvBNReLu(in_channel, out_channel, kernel_size):  # 基础的CBL结构，卷积、池化、激活
	return nn.Sequential(
		nn.Conv2d(in_channel, out_channel,
		          kernel_size = kernel_size,
		          stride = 1,
		          padding = kernel_size // 2),
		nn.BatchNorm2d(out_channel),
		nn.ReLU()
	)


class BaseInception(nn.Module):  # 基本的Inception模块
	def __init__(self, in_channel, out_channel_list, reduce_channel_list):
		super(BaseInception, self).__init__()  # 。
		self.branch1_conv = ConvBNReLu(in_channel,
		                               out_channel_list[0],
		                               kernel_size = 1)

		self.branch2_conv1 = ConvBNReLu(in_channel,  # 1*1卷积核进行压缩
		                                reduce_channel_list[0],
		                                kernel_size = 1)
		self.branch2_conv2 = ConvBNReLu(reduce_channel_list[0],
		                                out_channel_list[1],
		                                kernel_size = 3)

		self.branch3_conv1 = ConvBNReLu(in_channel,
		                                reduce_channel_list[1],
		                                kernel_size = 1)
		self.branch3_conv2 = ConvBNReLu(reduce_channel_list[1],
		                                out_channel_list[2],
		                                kernel_size = 5)

		self.branch4_pool = nn.MaxPool2d(3,  # 第四个分支不进行下采样
		                                 1,
		                                 1)
		self.branch4_conv = ConvBNReLu(in_channel,
		                               out_channel_list[3],
		                               kernel_size = 3)

	def forward(self, x):
		x1 = self.branch1_conv(x)
		x2 = self.branch2_conv2(self.branch2_conv1(x))
		x3 = self.branch3_conv2(self.branch3_conv1(x))
		x4 = self.branch4_conv(self.branch4_pool(x))
		out = torch.cat([x1, x2, x3, x4], dim = 1)
		return out


class InceptionNet(nn.Module):  # Inception网络定义
	def __init__(self):
		super(InceptionNet, self).__init__()
		self.block1 = nn.Sequential(
			nn.Conv2d(3, 64,
			          3,
			          2,
			          1),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)

		self.block2 = nn.Sequential(
			nn.Conv2d(64, 128,
			          3,
			          2,
			          1),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)

		self.block3 = nn.Sequential(
			BaseInception(128,
			              out_channel_list = [64, 64, 64, 64],
			              reduce_channel_list = [16, 16]),
			nn.MaxPool2d(3, 2, 1)
		)

		self.block4 = nn.Sequential(
			BaseInception(256,
			              out_channel_list = [96, 96, 96, 96],
			              reduce_channel_list = [32, 32]),
			nn.MaxPool2d(3, 2, 1)
		)

		self.fc = nn.Linear(384, 10)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = F.avg_pool2d(x, 2)  # 将输出结果转换到1*1feature_map上
		x = x.view(x.size(0), -1)  # 将特征图的维度拉成一维
		out = self.fc(x)
		return out

# 调用网络的接口
def VggNet():
	return VggBase()
def ResNet():
	return Resnet(ResBlock)
def MobileNetv1_small():
	return MobileNet()
def InceptionSmall():
	return InceptionNet()




