# 解析CIFAR10数据集
import pickle


def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding = 'bytes')
	return dict


label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#train_list = glob.glob("D:\pychram_workspace\datasets\cifar-10-batches-py\data_batch_*")
train_list = glob.glob(r"D:\pychram_workspace\datasets\cifar-10-batches-py\test_batch*")
#save_path = "D:\pychram_workspace\datasets\cifar-10-batches-py\TRAIN"
save_path = "D:\pychram_workspace\datasets\cifar-10-batches-py\TEST"

for l in train_list:

	print(l)
	l_dict = unpickle(l)
	# print(l_dict)
	# print(l_dict.keys())

	for im_idx, im_data in enumerate(l_dict[b'data']):
		# print(im_idx, ":", im_data)
		# print
		im_label = l_dict[b'labels'][im_idx]
		im_name = l_dict[b'filenames'][im_idx]

		# print(im_label, im_name, im_data)

		im_label_name = label_name[im_label]
		im_data = np.reshape(im_data, [3, 32, 32])
		im_data = np.transpose(im_data, (1, 2, 0))

		# 显示图像
		# cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
		# cv2.waitKey(0)
		# plt.imshow(im_data)
		# plt.axis('on')  # 可选，关闭坐标轴
		# plt.show()

		if not os.path.exists("{}/{}".format(save_path, im_label_name)):
			os.makedirs("{}/{}".format(save_path, im_label_name))
		cv2.imwrite("{}/{}/{}".format(save_path, im_label_name, im_name.decode("utf-8")), im_data)
