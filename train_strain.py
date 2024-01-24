# train the constrastive learning 
import argparse, time, random
from functools import partial

from torch.utils.data import DataLoader

from data_loading import *
from model import *
from eval import *
from import_data import getStrainData

import torch
import multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F

from sklearn import metrics


import warnings
warnings.filterwarnings("ignore")

# 设置随机种子
def set_seed(s):
	random.seed(s) # 设置 Python 内置的随机数生成器的种子
	np.random.seed(s) # 设置 NumPy 的随机数生成器的种子
	torch.manual_seed(s) # 设置 PyTorch 的随机数生成器的种子

	torch.cuda.manual_seed_all(s) # 设置所有可用的 CUDA 设备的随机数生成器的种子
	# 这将确保在使用 PyTorch 进行随机操作时，每次运行代码时都会得到相同的随机结果。
	#add additional seed
	torch.backends.cudnn.deterministic=True # 设置使用 CuDNN 时的随机数生成器为确定性模式
	# CuDNN 是深度学习库的一部分，它可以提高深度神经网络的计算性能。通过将确定性模式设置为 True，可以确保在使用 CuDNN 时，每次运行代码时都会得到相同的计算结果。
	torch.use_deterministic_algorithms = True # 设置使用确定性算法
	# 确保在使用一些特定的算法（如卷积算法）时，每次运行代码时都会得到相同的结果，从而增加结果的可重复性




def train(dl_model, model_path, kmer, margin, batch_size, lr, epoch, filename, device="cuda:1", verbose=True,num_workers=32):
	# 加载数据集阶段
	## data loading phase
	if verbose:
		print(" |- Start preparing dataset...")
	start_dataload = time.time()
	strainPhage, strainPhageDNA, strainHost, strainHostDNA, strainLabel = getStrainData(kmer)
	# strain数据共12480条

	# 获得list 训练加验证宿主的label共187
	print("	|-* Provided training sets totally has [", len(strainLabel),"] labels.")

	# 划分训练和验证集
	strain_train_dataset = strain_fasta_dataset(strainPhage[:9984], strainPhageDNA[:9984], strainHost[:9984], strainHostDNA[:9984], strainLabel[:9984])
	strain_valid_dataset = strain_fasta_dataset(strainPhage[9984:11232], strainPhageDNA[9984:11232], strainHost[9984:11232],
												strainHostDNA[9984:11232], strainLabel[9984:11232])
	# 获得训练数据集
	'''
	DataLoader组合了数据集（dataset） + 采样器(sampler)
	然后用collate_fn来把它们打包成batch
	l2fa:{'host的label','FCGR'} FCGR是64X64的2D数组
	collate_fn：如何取样本的
	'''
	train_generator = DataLoader(strain_train_dataset, batch_size, collate_fn=partial(my_collate_strain), num_workers=num_workers)
    # 验证数据集 131个噬菌体
	valid_generator = DataLoader(strain_valid_dataset, batch_size, collate_fn=partial(my_collate_strain), num_workers=num_workers)


	# cached the data set
	# 共分成了37个组数据，除了最后一组，每组数据的类型[5984,1,64,64] 5984=32*187（host数量）每批32个phage，每个phage可能与187个host有侵染关系
	# cached_train_ph[5984,1,64,64]
	# cached_train_bt[5984,1,64,64]
	# cached_train_label[5984]
	cached_train_ph, cached_train_bt, cached_train_label = [], [], []
	cached_valid_ph, cached_valid_bt, cached_valid_label = [], [], []

	## loading the image pairs for constrastive training
	# phs, bts都是以FCGR表示的
	for phs, bts, labels in train_generator:
		#X = torch.tensor(X, dtype = torch.float32).transpose(1,2)
		imgs_ph = torch.tensor(phs, dtype = torch.float32)
		imgs_bt = torch.tensor(bts, dtype = torch.float32)
		
		cached_train_ph.append(torch.unsqueeze(imgs_ph, dim=1)) # 增加维度1，通道数
		cached_train_bt.append(torch.unsqueeze(imgs_bt, dim=1))
		cached_train_label.append(torch.tensor(labels))
	# 等待改正 test
	for phs, bts, labels in valid_generator:
		# X = torch.tensor(X, dtype = torch.float32).transpose(1,2)
		imgs_ph = torch.tensor(phs, dtype=torch.float32)
		imgs_bt = torch.tensor(bts, dtype=torch.float32)

		cached_valid_ph.append(torch.unsqueeze(imgs_ph, dim=1))  # 增加维度1，通道数
		cached_valid_bt.append(torch.unsqueeze(imgs_bt, dim=1))
		cached_valid_label.append(torch.tensor(labels))
	
	print(" |- loading [ok].")
	used_dataload = time.time() - start_dataload
	print("  |-@ used time:", round(used_dataload,2), "s")

	start_train = time.time()

	# model 2 (using CNN module)
	if dl_model == "CNN":
		model = cnn_module().to(device)
		optimizer = optim.Adam(model.parameters(), lr=lr)

	criterion = ContrastiveLoss(margin) 

	if verbose:
		print(f" |- Total number of {dl_model} has parameters %d:" %(sum([p.nelement() for p in model.parameters()])))
		print("  |- Training started ...")

	# start training
	# 用于保存每轮损失，绘制loss function曲线
	epoch_loss_list = []
	epoch_acc_valid, epoch_acc_test, epoch_cm = [], [], []
	current_best_valid_acc = -100
	for ep in range(epoch):
		epoch_loss = 0
		for i in range(len(cached_train_ph)):
			# 准备数据
			phs, bts, labels = cached_train_ph[i], cached_train_bt[i], cached_train_label[i]

			phs = phs.to(device)
			bts = bts.to(device)
			labels = labels.to(device)
			# 获得嵌入
			# [5984,1,64,64]
			embed_ph = model(phs)
			embed_bt = model(bts)
			# 计算损失
			loss = criterion(embed_ph, embed_bt, labels)
			epoch_loss += loss.item()
			# 优化参数
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		# 打印这轮损失
		epoch_loss_list.append(epoch_loss)
		print("Epoch-%d, Loss=%f" %(ep,epoch_loss))
		#
		acc_valid, _, _ = testStrain(model, cached_valid_ph, cached_valid_bt, cached_valid_label, device, 1, True)
		epoch_acc_valid.append(acc_valid)
		# 保存最佳模型
		if acc_valid > current_best_valid_acc: # to be consistent with the following one. 
			current_best_valid_acc = acc_valid
			torch.save(model.state_dict(), model_path)

	# 保存损失
	saveLoss(epoch_loss_list, filename)
	idx = epoch_acc_valid.index(max(epoch_acc_valid))
	print(f"[Valid epoch idx/epoch]:{idx}/{epoch}, [valid acc]:{epoch_acc_valid[idx]},[valid average acc]:{sum(epoch_acc_valid)/len(epoch_acc_valid)}")
	used_train = time.time() - start_train
	print(" @ used training time:", round(used_train,2), "s. Total time:", round(used_train+used_dataload,2))



if __name__ == "__main__":

	set_seed(123)

	# 参数设置
	lr = 1e-3
	epoch = 200
	batch_size = 256 # 由于数据稀少所有每批次大小为256
	margin = 1
	kmer = 6
	model_save_path = "model_save/"
	model_info = "cherry_cnn2_model_margin-{}-epoch-{}".format(margin, epoch)
	model_dir = model_save_path + model_info
	model = "CNN"
	filename = 'cherryCNN2-epoch-{}.txt'.format(epoch)
	# model train
	train("CNN", model_dir, kmer, margin, batch_size, lr, epoch, filename)
