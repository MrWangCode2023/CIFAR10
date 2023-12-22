import torch
import torch.nn as nn
import torchvision
#from torch_net import pytorch_resnet18
from Net_common import VggNet, ResNet, MobileNetv1_small, InceptionSmall
from load_cifar10 import train_data_loader, test_data_loader
import os
import tensorboardX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否使用GPU
epoch_num = 200
lr = 0.01
batch_size = 128
net = ResNet() # 网络接口
net_name = net.__class__.__name__  # 网络的名称
net = net.to(device)
loss_func = nn.CrossEntropyLoss()  # loss
optimizer = torch.optim.Adam(net.parameters(), lr= lr)  # optimizer
#optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size = 1,
                                            gamma = 0.9) # 学习率衰减

model_path = f"models\\{net_name}"
log_path = f"logs\\{net_name}"
if not os.path.exists(log_path):
	os.makedirs(log_path)
if not os.path.exists(model_path):
	os.makedirs(model_path)
writer = tensorboardX.SummaryWriter(log_path)

step_n = 0 # 整体的step计数器，作为记录log的输入参数
for epoch in range(epoch_num):  # 训练过程
	print("epoch is ", epoch)
	net.train()  # train BN dropout,网络进入训练模式

	for i, data in enumerate(train_data_loader): # 遍历数据
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device) # 从数据集拿到数据

		outputs = net(inputs) # 将拿到的数据送入网络
		loss = loss_func(outputs, labels)
		optimizer.zero_grad() # 梯度清零
		loss.backward()
		optimizer.step() # 更新参数

		_, pred = torch.max(outputs.data, dim=1) # 对data  数据的第二维度进行求最值,最高得分作为预测结果

		correct = pred.eq(labels.data).cpu().sum() # 对预测结果pred和labels的数据（标签）进行比较是否相同,得到预测正确的样本数量
		writer.add_scalar("train loss", loss.item(), global_step = step_n)
		writer.add_scalar("train correct",
		                  100.0 * correct.item() / batch_size, global_step = step_n)

		im = torchvision.utils.make_grid(inputs)
		writer.add_image("train im", im, global_step = step_n)

		step_n += 1

	torch.save(net.state_dict(), "{}\{}.pth".format(model_path,
	                                                epoch+1)) #存放当前模型中的参数
	scheduler.step() #每个epoch更新学习率

	sum_loss = 0
	sum_correct = 0
	for i, data in enumerate(test_data_loader): # 遍历数据，测试脚本
		net.eval()  # train BN dropout
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)  # 从数据集拿到数据

		outputs = net(inputs)  # 将拿到的数据送入网络
		loss = loss_func(outputs, labels)
		_, pred = torch.max(outputs.data, dim = 1)  # 对data数据的第二维度进行求最值,最高得分作为预测结果
		correct = pred.eq(labels.data).cpu().sum()  # 对预测结果pred和labels的数据（标签）进行比较是否相同,得到预测正确的样本数量
		sum_loss += loss.item()
		sum_correct += correct.item()
		im = torchvision.utils.make_grid(inputs)
		writer.add_image("test im", im, global_step = step_n)
		# writer.add_scalar("test correct",
		#                   100.0 * correct / batch_size, global_step = step_n)

	# print(sum_correct * 1.0, len(test_data_loader))
	test_loss = sum_loss * 1.0 / len(test_data_loader)
	test_correct = sum_correct * 100.0 / len(test_data_loader) / batch_size

	writer.add_scalar("test loss", loss.item(), global_step = epoch + 1)
	writer.add_scalar("test correct",
	                  100.0 * correct.item() / batch_size, global_step = epoch + 1)

	print("epoch is", epoch + 1, "loss is :", test_loss,
	       "test correct is:", test_correct)


writer.close()











