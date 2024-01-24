# train the constrastive learning 
import argparse, time, random
from functools import partial

from torch.utils.data import DataLoader

from data_loading import *
from model import *
from eval import *

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

'''
训练模型
dl_model：要训练的深度学习模型
data_set：用于训练的数据集
model_path：保存训练模型的路径
margin：边界值（margin），在对比学习中用于控制同类样本和异类样本之间的距离
batch_size：训练时使用的批次大小，表示每次迭代中用于更新模型的样本数量
lr：学习率（learning rate），用于控制模型参数更新的步长
epoch：训练的总轮数，表示将整个数据集用于训练的次数
device：指定训练所使用的设备，例如 "cuda:0" 表示使用第一个 CUDA 设备进行训练，默认为 "cuda:0"
verbose：布尔值，控制是否输出训练过程中的详细信息，默认为 True。
'''
def train(dl_model, data_set, model_path, kmer, margin, batch_size, lr, epoch, filename,\
	device="cuda:1", num_workers=32, verbose=True):

	cores = mp.cpu_count() # 获取计算机的 CPU 核心数
	if num_workers > 0 and num_workers < cores: # 如果指定的工作进程数量大于0且小于计算机的 CPU 核心数，则将工作进程数量设置为指定的值
		cores = num_workers

	# 加载数据集阶段
	## data loading phase
	if verbose:
		print(" |- Start preparing dataset...")
	# 这个是传入的数据
	# data_set = [host_fa, host_list, train_phage_fa, train_host_gold, valid_phage_fa, valid_host_gold]
	# 数据集地址
	host_fa_file, spiece_file, phage_train_file, host_train_file, phage_valid_file, host_valid_file = data_set
	
	start_dataload = time.time()

	# 获得list 训练加验证宿主的label共187
	train_data_labels = get_data_host_sets([host_train_file, host_valid_file])
	print("	|-* Provided training sets totally has [", len(train_data_labels),"] hosts.")
	'''
	获得一个继承Dataset的fa_train_dataset
	return: phage name, phage DNA, host id
	'''
	fa_train_dataset = fasta_dataset(phage_train_file, spiece_file, host_train_file)
	#223个宿主  l2fa:{'host的label','FCGR'} FCGR是64X64的2D数组
	l2fa = get_host_fa(fa_train_dataset.get_s2l_dic(), host_fa_file, kmer)
	# add filters for host label here 包括验证集和测试集共187个 每个宿主64X64 187个宿主的kmer FCGR表示
	l2fa_filter = get_host_fa(fa_train_dataset.get_s2l_dic(), host_fa_file, kmer, train_data_labels)
	print("	|-[!] Checking host label information filtering for the training [non_filter:", len(l2fa.keys()), ", filtered:", len(l2fa_filter.keys()), "].")
	# 获得训练数据集
	'''
	DataLoader组合了数据集（dataset） + 采样器(sampler)
	然后用collate_fn来把它们打包成batch
	l2fa:{'host的label','FCGR'} FCGR是64X64的2D数组
	collate_fn：如何取样本的
	'''
	train_generator = DataLoader(fa_train_dataset, batch_size, collate_fn=partial(my_collate_fn, kmer=kmer, l2fa=l2fa_filter), num_workers=num_workers) 
    # 验证数据集 131个噬菌体
	fa_valid_dataset = fasta_dataset(phage_valid_file, spiece_file, host_valid_file)
	valid_generator = DataLoader(fa_valid_dataset, batch_size, collate_fn=partial(my_collate_fn2, kmer=kmer), num_workers=num_workers) 

	# cached the data set
	# 共分成了37个组数据，除了最后一组，每组数据的类型[5984,1,64,64] 5984=32*187（host数量）每批32个phage，每个phage可能与187个host有侵染关系
	# cached_train_ph[5984,1,64,64]
	# cached_train_bt[5984,1,64,64]
	# cached_train_label[5984]
	cached_train_ph, cached_train_bt, cached_train_label = [], [], []
	cached_valid_ph, cached_valid_label, cached_valid_phageName = [], [], []

	## loading the image pairs for constrastive training
	# phs, bts都是以FCGR表示的
	for phs, bts, labels in train_generator:
		#X = torch.tensor(X, dtype = torch.float32).transpose(1,2)
		imgs_ph = torch.tensor(phs, dtype = torch.float32)
		imgs_bt = torch.tensor(bts, dtype = torch.float32)
		
		cached_train_ph.append(torch.unsqueeze(imgs_ph, dim=1)) # 增加维度1，通道数
		cached_train_bt.append(torch.unsqueeze(imgs_bt, dim=1))
		cached_train_label.append(torch.tensor(labels))

	for phs, labels, phName in valid_generator:
		imgs_ph = torch.tensor(phs, dtype = torch.float32)
		cached_valid_ph.append(torch.unsqueeze(imgs_ph, dim=1))
		cached_valid_label.append(torch.tensor(labels))
		cached_valid_phageName.append(phName)
	
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
	epoch_loss_list = []
	# start training
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
		acc_valid, _, _ = test(model, cached_valid_ph, l2fa_filter, cached_valid_label, device, 1, True)
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

	# parser = argparse.ArgumentParser(description='<Contrastive learning for the phage-host identification>')
	#
	# parser.add_argument('--model',       default="CNN", type=str, required=True, help='contrastive learning encoding model')
	# parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
	# parser.add_argument('--device',       default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')
	#
	# parser.add_argument('--kmer',       default=5,       type=int, required=True, help='kmer length')
	# parser.add_argument('--margin',     default=1,       type=int, required=True, help='Margins used in the contrastive training')
	# parser.add_argument('--lr',     	default=1e-3,   type=float, required=False, help='Learning rate')
	# parser.add_argument('--epoch',      default=20,       type=int, required=False, help='Training epcohs')
	# parser.add_argument('--batch_size' ,default=64,      type=int,  required=False, help="batch_size of the training.")
	# parser.add_argument('--workers',     default=64,       type=int, required=False, help='number of worker for data loading')
	#
	# # data related input
	# parser.add_argument('--host_fa',   default="",  type=str, required=True, help='Host fasta files')
	# parser.add_argument('--host_list', default="",  type=str, required=True, help='Host species list')
	#
	# parser.add_argument('--train_phage_fa', default="",   type=str, required=True, help='Trainset Phage fasta file')
	# parser.add_argument('--train_host_gold', default="",  type=str, required=True, help='Trainset Phage infectable host label')
	# parser.add_argument('--valid_phage_fa', default="",   type=str, required=True, help='Validset Phage fasta file')
	# parser.add_argument('--valid_host_gold', default="",  type=str, required=True, help='Validset Phage infectable host label')
	#
	#
	# args = parser.parse_args()
	#
	# data_set=[args.host_fa, args.host_list, args.train_phage_fa, args.train_host_gold, args.valid_phage_fa, args.valid_host_gold]
	#
	# # model train
	# train(args.model, data_set, args.model_dir, args.kmer,args.margin, args.batch_size, \
	# 	args.lr, args.epoch, args.device, args.workers)
	# 参数设置
	lr = 1e-3
	epoch = 150
	batch_size = 32
	margin = 1
	kmer = 6
	model_save_path = "model_save/"
	model_info = "cherryCNN2-attention_model_margin-{}-epoch-{}.pth".format(margin, epoch)
	model_dir = model_save_path + model_info
	model = "CNN"

	# host data
	#cherry
	host_fa = "data/CHERRY_benchmark_datasplit/cherry_host.fasta" # 全部host的fasta 223个
	host_list = "data/CHERRY_benchmark_datasplit/species.txt" # 全部host的名字 223个
	#deephost
	# host_fa = "data/deepHOST_benchmark_datasplit/DeepHost_host_117.fasta" # 全部host的fasta 223个
	# host_list = "data/deepHOST_benchmark_datasplit/species_117.txt" # 全部host的名字 223个
			# Train	Validation	Test
	# phage	1175		131		634
	# host	182			52		95 (59+36 unseen)
	# phage data
	# cherry
	train_phage_fa = "data/CHERRY_benchmark_datasplit/CHERRY_train.fasta"	# train phage fasta
	train_host_gold = "data/CHERRY_benchmark_datasplit/CHERRY_y_train.csv" # 1175个phage对应的host name
	valid_phage_fa = "data/CHERRY_benchmark_datasplit/CHERRY_val.fasta" # valid phage fasta
	valid_host_gold = "data/CHERRY_benchmark_datasplit/CHERRY_y_val.csv" # 131个valid host name
	# deephost
	# train_phage_fa = "data/deepHOST_benchmark_datasplit/DeepHost_train.fasta"
	# train_host_gold = "data/deepHOST_benchmark_datasplit/DeepHost_y_train.csv"
	# valid_phage_fa = "data/deepHOST_benchmark_datasplit/DeepHost_val.fasta"
	# valid_host_gold = "data/deepHOST_benchmark_datasplit/DeepHost_y_val.csv"
	filename = 'cherryCNN2-attention-epoch-{}.txt'.format(epoch)
	data_set = [host_fa, host_list, train_phage_fa, train_host_gold, valid_phage_fa, valid_host_gold]
	# model train
	train("CNN", data_set, model_dir, kmer, margin, batch_size, lr, epoch, filename)
