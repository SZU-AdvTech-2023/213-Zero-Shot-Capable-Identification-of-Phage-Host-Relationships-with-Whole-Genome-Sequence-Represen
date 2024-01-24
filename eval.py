# prediction 
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader

from data_loading import *
from model import *

import numpy as np
from sklearn import metrics

# 验证结果
def testStrain(model, cached_valid_ph, cached_valid_bt, cached_valid_label, device, threshold=1, verbose=False):
	# 预测的label，预测的距离，真实标签
	pred_label, gold_list = [], []
	total_batch = len(cached_valid_ph)
	with torch.no_grad():
		# 对于每个批次
		# 获得phs，host，label
		for i in range(total_batch):
			phs, hosts, labels = cached_valid_ph[i], cached_valid_bt[i], cached_valid_label[i]
			phs = phs.to(device)
			hosts = hosts.to(device)
			labels = labels.to(device)

			embed_phs = model(phs)
			embed_host = model(hosts)

			# 计算距离
			diff = embed_phs - embed_host
			dist_sq = torch.sum(torch.pow(diff, 2), 1)
			dist = torch.sqrt(dist_sq)
			# 这个得到的是每个批次的tensor，其中大于等于1的为0，小于1的为1
			pred_label.extend(torch.where(dist.to("cpu") >= threshold, torch.tensor(0), torch.tensor(1)).detach().numpy())
			# 真实标签
			gold_list.extend(labels.to("cpu").detach().numpy())

	acc = metrics.accuracy_score(gold_list, pred_label)
	if verbose:
		print(f"@ Given set  Accuracy is {acc}")

	return acc, pred_label, gold_list


def test(model, cached_ph, l2fa, cached_label, device, threshold=1, verbose=False):

	# first generate embeddings for the host.
	host_vec = np.array([l2fa[l] for l in l2fa.keys()])
	host_vec = torch.tensor(host_vec, dtype=torch.float32).to(device)
	host_vec = torch.unsqueeze(host_vec, dim=1)

	embed_bts = model(host_vec)
	label_list = list(l2fa.keys())

	pred_list, pred_dist, gold_list = [], [], []
	total_batch = len(cached_ph)

	with torch.no_grad():
		for i in range(total_batch):
			phs, labels = cached_ph[i], cached_label[i]

			phs = phs.to(device)
			labels = labels.to(device)
			embed_phs = model(phs)

			# local calcuation of the distance scores
			for e_ph in embed_phs:
				diff =  embed_bts - e_ph
				dist_sq = torch.sum(torch.pow(diff, 2), 1)
				dist = torch.sqrt(dist_sq)

				pred_dist.append(dist.to("cpu").detach().numpy())
				idx = torch.argmin(dist).to("cpu").detach().numpy()
				
				pred_list.extend([label_list[idx]])
				
			gold_list.extend(labels.to("cpu").detach().numpy())

	acc = metrics.accuracy_score(gold_list, pred_list)
	if verbose:
		print(f"@ Given set  Accuracy is {acc}")

	return acc, pred_dist, gold_list


# prediction without provide gold standard
def predict(model, cached_ph, l2fa, device):

	# first generate embeddings for the host.
	host_vec = np.array([l2fa[l] for l in l2fa.keys()])
	host_vec = torch.tensor(host_vec, dtype=torch.float32).to(device)
	host_vec = torch.unsqueeze(host_vec, dim=1)

	embed_bts = model(host_vec)
	label_list = list(l2fa.keys())

	pred_list, pred_dist = [], []
	total_batch = len(cached_ph)

	with torch.no_grad():
		for i in range(total_batch):
			phs = cached_ph[i]
			phs = phs.to(device)
			embed_phs = model(phs)

			# local calcuation of the distance scores
			for e_ph in embed_phs:
				diff =  embed_bts - e_ph
				dist_sq = torch.sum(torch.pow(diff, 2), 1)
				dist = torch.sqrt(dist_sq)

				pred_dist.append(dist.to("cpu").detach().numpy())

				#idx = torch.argmin(dist).to("cpu").detach().numpy()	
				#pred_list.extend([label_list[idx]])
				
	return pred_dist


if __name__ == "__main__":

	host_fa = "data/CHERRY_benchmark_datasplit/cherry_host.fasta"
	host_list = "data/CHERRY_benchmark_datasplit/species.txt"

	# test data
	test_phage_fa = "data/CHERRY_benchmark_datasplit/CHERRY_test.fasta"
	test_host_gold = "data/CHERRY_benchmark_datasplit/CHERRY_y_test.csv"

	# host_fa = "data/deepHOST_benchmark_datasplit/DeepHost_host_117.fasta"
	# host_list = "data/deepHOST_benchmark_datasplit/species_117.txt"

	# test data
	# test_phage_fa = "data/deepHOST_benchmark_datasplit/DeepHost_test.fasta"
	# test_host_gold = "data/deepHOST_benchmark_datasplit/DeepHost_y_test.csv"

	model_dir = "model_save/cherry/cherryCNN2-attention_model_margin-1-epoch-150.pth"
	device = "cuda:1"
	kmer = 6
	num_workers = 32
	batch_size = 32

	# preparing the test data for the evaluation.
	## 1. loading model
	print("@ Loading model ... ", end="")
	## parparing host data information.
	model = cnn_module()
	
	model.load_state_dict(torch.load(model_dir))
	model = model.to(device)

	print("[ok]")

	## 2. loading data
	print("@ Loading phage dataset ... ", end="")
	spiece_file = host_list
	host_fa_file = host_fa
	phage_test_file = test_phage_fa
	host_test_file = test_host_gold

	fa_test_dataset = fasta_dataset(phage_test_file, spiece_file, host_test_file)
	test_generator = DataLoader(fa_test_dataset, batch_size, collate_fn=partial(my_collate_fn2, kmer=kmer), num_workers=num_workers) 

	cached_test_ph,  cached_test_label, test_phName = [], [], []

	for phs, labels, phName in test_generator:
		imgs_ph = torch.tensor(phs, dtype = torch.float32)
		cached_test_ph.append(torch.unsqueeze(imgs_ph, dim=1))
		cached_test_label.append(torch.tensor(labels))
		test_phName.extend(phName)

	print("[ok]")

	# viualizaiton
	print("@ Loading Host dataset ... ", end="")
	s2l_dic = fa_test_dataset.get_s2l_dic()
	l2fa = get_host_fa(s2l_dic, host_fa_file, kmer)
	l2sn = fa_test_dataset.get_l2s_dic()
	label_list = list(l2fa.keys())
	
	print("[ok]")
	print("@ Start prediction ...")
	if test_host_gold != "":
		acc_test, host_pred_list, gold_list  = test(model, cached_test_ph, l2fa, cached_test_label, device)
		print(f"cnn2attention[Test acc]:{acc_test}")
	else:
		host_pred_list = predict(model, cached_test_ph, l2fa, device)

	# for i in range(len(cached_test_ph)):
	#
	# 	print(test_phName[i], end="\t")
	# 	idxs = np.argsort(host_pred_list[i])
	# 	for idx in idxs:
	# 		print(l2sn[idx]+"_"+str(host_pred_list[i][idx]), end=" ")
	#
	# 	print("")

