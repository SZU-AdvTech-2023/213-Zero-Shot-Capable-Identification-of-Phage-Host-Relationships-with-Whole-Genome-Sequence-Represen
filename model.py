# preparing for the constrastive learning

import torch
from torch import nn
# 对比损失
class ContrastiveLoss(torch.nn.Module):
	def __init__(self, margin=1.0):
		super(ContrastiveLoss, self).__init__()
		# 临界值
		self.margin = margin

	def forward(self, x0, x1, y):
		# 计算两个样本之间的差异
		diff = x0 - x1
		# 计算差异的平方和
		dist_sq = torch.sum(torch.pow(diff, 2), 1)
		# 计算欧氏距离
		dist = torch.sqrt(dist_sq)
		# 计算差异与设定的边界值之间的差值
		mdist = self.margin - dist
		# 将差值限制在不小于0的范围内
		dist = torch.clamp(mdist, min=0.0)
		# 计算损失
		loss = y * dist_sq + (1-y) * torch.pow(dist,2)
		# 对损失进行求和并除以样本数和2
		loss = torch.sum(loss) / 2.0 / x0.size()[0]

		return loss


def distance(x1, x2, dist_type="euc"):

	if dist_type == "euc":
		dist = torch.cdist(x1,x2)**2

	if dist_type == "cos":
		cos = nn.CosineSimilarity(dim=1, eps=1e-6)
		dist = cos(x1, x2)

	return dist


# cnn2
# 定义编码器 x的shape[5984,1,64,64]
class cnn_module_3conv(nn.Module):
	def __init__(self, kernel_size=3, dr=0):
		super(cnn_module_3conv, self).__init__()
		# 第一层卷积层，输入通道数，输出通道数（特征图数），卷积核大小，步长
		self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=kernel_size, stride=2)
		# 批归一化层
		self.bn1 = nn.BatchNorm2d(64)
		# 第二层卷积层，输入通道数，输出通道数（特征图数），卷积核大小，步长
		self.conv2 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=kernel_size, stride=2)
		# 批归一化层
		self.bn2 = nn.BatchNorm2d(128)
		# 第三层卷积层，输入通道数，输出通道数（特征图数），卷积核大小，步长
		self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=2)
		# 批归一化层
		self.bn3 = nn.BatchNorm2d(256)
		# 激活函数
		self.relu = nn.ReLU()
		# 最大池化层，池化核大小
		self.maxpool = nn.MaxPool2d(2)
		# Dropout层，用于随机失活一部分神经元，防止过拟合
		self.dropout = nn.Dropout(dr)
		# 全连接层，输入大小为4608，输出大小为512
		self.fc1 = nn.Linear(2304, 512) #

	def forward(self, x):
		# [5984, 1, 64, 64]
		# print("initial",end='')
		# print(x.shape)
		# 前向传播函数
		# 输入x经过卷积、批归一化、ReLU激活函数
		# [5984, 64, 29, 29]
		x = self.bn1(self.relu(self.conv1(x)))
		# print("cn1",end='')
		# print(x.shape)
		# [5984, 128, 12, 12]
		# 输入x经过卷积、批归一化、ReLU激活函数
		x = self.bn2(self.relu(self.conv2(x)))
		# print("cn2", end='')
		# print(x.shape)
		# 输入x经过卷积、批归一化、ReLU激活函数
		x = self.bn3(self.relu(self.conv3(x)))
		# print("cn3", end='')
		# print(x.shape)
		# [5984, 128, 6, 6]
		# 输入x经过最大池化层
		x = self.maxpool(x)
		# print("max", end='')
		# print(x.shape)
		# 将x展平为一维向量，经过全连接层
		x = self.fc1(torch.flatten(x, 1))
		# print("fc1",end='')
		# print(x.shape)
		return x

# cnn2
# define the basic module
class cnn_module(nn.Module):
	def __init__(self, kernel_size=7, dr=0):
		super(cnn_module, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=2)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu = nn.ReLU()

		self.maxpool = nn.MaxPool2d(2)
		self.dropout = nn.Dropout(dr)

		self.fc1 = nn.Linear(4608, 512)
		#  注意力机制测试
		self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=1)

	def forward(self, x):
		x = self.bn1(self.relu(self.conv1(x)))
		x = self.bn2(self.relu(self.conv2(x)))
		x = self.maxpool(x)

		x = self.fc1(torch.flatten(x, 1))
		# 添加注意力机制
		x = x.unsqueeze(0)  # 增加时间维度
		x, _ = self.attention(x, x, x)
		x = x.squeeze(0)  # 移除时间维度

		return x