# load_cifar10
import glob
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

label_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
label_dict = {}
for idx, name in enumerate(label_name):
	label_dict[name] = idx


def default_loader(path):
	return Image.open(path).convert("RGB")  # 读取图像数据，转换成RGB形式


train_transform = transforms.Compose([
	transforms.RandomResizedCrop((28, 28)),
	transforms.RandomHorizontalFlip(),  # 水平翻转
	# transforms.RandomVerticalFlip(),  # 垂直翻转
	# transforms.RandomRotation(90),  # 图像以最多90度的角度随机旋转
	# transforms.RandomGrayscale(0.1),
	# transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
	transforms.ToTensor(),
	transforms.Normalize(
		(0.4914, 0.4822, 0.4465),
		(0.2023, 0.1994, 0.2010)
	)
])

test_transform = transforms.Compose([
	transforms.RandomResizedCrop((28, 28)),
	# transforms.RandomHorizontalFlip(),  # 水平翻转
	# transforms.RandomVerticalFlip(),  # 垂直翻转
	# transforms.RandomRotation(90),  # 图像以最多90度的角度随机旋转
	# transforms.RandomGrayscale(0.1),
	# transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
	transforms.ToTensor(),
	transforms.Normalize(
		(0.4914, 0.4822, 0.4465),
		(0.2023, 0.1994, 0.2010)
	)
])


class MyDataset(Dataset):
	def __init__(self, im_list,
	             transform = None,
	             loader = default_loader):
		super(MyDataset, self).__init__()

		imgs = []
		for im_item in im_list:  # im_list的元素：“/kaggle/working/cifar10wha/TRAIN/frog/bufo_s_001873.png”
			im_label_name = im_item.split("\\")[-2]
			imgs.append([im_item, label_dict[im_label_name]])  # 将图像路径和对应的标签进行配对，存储在数组中

		self.imgs = imgs
		self.transform = transform
		self.loader = loader

	def __getitem__(self, index):
		im_path, im_label = self.imgs[index]
		im_data = self.loader(im_path)  # 已经读取到图像
		if self.transform is not None:
			im_data = self.transform(im_data)

		return im_data, im_label

	def __len__(self):
		return len(self.imgs)

# D:\pychram_workspace\datasets\cifar-10-batches-py\CIFAR10_decoded\TRAIN\airplane\
im_train_list = glob.glob("D:\pychram_workspace\datasets\cifar-10-batches-py\CIFAR10_decoded\TRAIN\*\*.png")
im_test_list = glob.glob("D:\pychram_workspace\datasets\cifar-10-batches-py\CIFAR10_decoded\TEST\*\*.png")

train_dataset = MyDataset(im_train_list, transform = train_transform)
test_dataset = MyDataset(im_test_list, transform = transforms.ToTensor())

train_data_loader = DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True, num_workers = 0)
test_data_loader = DataLoader(dataset = test_dataset, batch_size = 128, shuffle = False, num_workers = 0)

print("num of train:", len(train_dataset))
print("num of test:", len(test_dataset))