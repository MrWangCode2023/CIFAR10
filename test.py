import torch
import glob
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from Net_common import VggNet, ResNet, MobileNetv1_small, InceptionSmall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ResNet() # 选择网络
ResNet.load_state_dict(torch.load("模型参数路径")) # 加载训练好的模型权重
net.to(device) # 将加载好权重的网络放到device上
im_path_list = glob.glob(r"D:\pychram_workspace\datasets\cifar-10-batches-py\CIFAR10_decoded\TEST\*\*") # 获取测试集图片路径列表
np.random.shuffle(im_path_list) # 将，上面的图像列表打乱
label_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

test_transform = transforms.Compose([
	transforms.RandomResizedCrop((28, 28)),
	transforms.ToTensor(),
	transforms.Normalize(
		(0.4914, 0.4822, 0.4465),
		(0.2023, 0.1994, 0.2010)
	)
]) #定义图像预处理方法

for im_patn in im_path_list:
	net.eval() # 将模型设置为评估模式，关闭BN层，相当于锁定模型参数不在更新
	im_data = Image.open(im_patn) # PIL读取图像,RGB格式
	inputs = test_transform(im_data) # 预处理图像
	inputs = torch.unsqueeze(inputs, dim = 0) #扩充数据维度(C,W,H)-->(N,C,W,H)
	inputs = inputs.to(device)
	outputs = net.forward(inputs)

	_, pred = torch.max(outputs.data, dim = 1)
	print(label_name[pred.cpu().numpy()[0]])
	img = np.asfarray(im_data) # np.asfarray() 是 NumPy 库中的一个函数，用于将输入数据转换为一个具有浮点数类型的 NumPy 数组。
	img = img[:, :, [2, 1, 0]] # CV读取的格式BGR，需要转换格式
	cv2.imshow("im", img)
	cv2.waitKey()



