# data loading for the paired samples


from pyfaidx import Fasta
from torch.utils.data import Dataset

from fasta2CGR import *
import numpy as np

# 训练集和验证集中host可能重复，这个用于去重，得到list labels ：host名字
def get_data_host_sets(file_name_list):

	labels = []

	for fn in file_name_list:
		s_in = open(fn)
		
		for line in s_in:
			line_info = line.strip("\n")
			labels.append(line_info)

		s_in.close()

	return list(set(labels))

'''
输入host文件，获得spieve_dic:{'name':'label'}
host:label2species:['name']
'''
def get_label_map(species_file):
    s_in = open(species_file)
    spiece_dic, label2species = {}, []

    for idx, line in enumerate(s_in):
        line_info = line.strip("\n")
        specie_label = line_info.split("\t")[1]
        spiece_dic[specie_label] = idx
        label2species.append(specie_label)

    s_in.close()

    return spiece_dic, label2species

'''
为每个phage对上相应host的label(编号)
这里的host_file中的每个host都一一对应某个噬菌体
label=['','','','','','']编号
'''
def load_host_label(host_file, l2s_dic):
	if host_file == '':
		return []

	s_in = open(host_file)
	labels = []

	for line in s_in:
		line_info = line.strip("\n")
		labels.append(l2s_dic[line_info])

	s_in.close()

	return labels

# get all host representation of the host_fa_file according to the keepList.
'''
s2l_dic:{'host name':'label'}
host_fa_file:host fasta文件
return l2fa:{'host的label','FCGR'} FCGR是64X64的2D数组
'''
def get_host_fa(s2l_dic, host_fa_file, kmer, keep_list=[]):

	l2fa = {}
	# loading host fa information into 
	wgs = Fasta(host_fa_file)
	for bn in wgs.keys():
		# filtering labels not in the keep_list
		check = bn.replace("_", " ")
		if len(keep_list) > 0 and check not in keep_list:
			continue

		seq = wgs[bn][:].seq
		# 计算给定seq的kmer的频率
		fc = count_kmers(seq, kmer)
		# 计算给定seq和fc的kmer概率   即每个k-mer的频率换成k-mer概率
		f_prob = probabilities(seq, fc, kmer)
		# 获得每个DNA的FCGR表示
		chaos_k = chaos_game_representation(f_prob, kmer)

		label = s2l_dic[bn.replace("_", " ")]
		l2fa[label] = chaos_k

	return l2fa


def my_collate_fn(batch, kmer, l2fa):

	# images表示每个phage的FCGR表示
	# hosts表示可能发生侵染关系的host FGCR
	# label表示image-host之间是否有相互作用
	images, hosts, labels = [],[],[]
	# 对于每批32个phage，对于训练的187的host都要测一下
	# 这样的话每个批次的phage-host-label  都变成了32*187=5984  （除了最后一个批次）
	for name, seq, label in batch:
		phage_name = name
		seq = seq

		# FCGR, represntation for phage
		fc = count_kmers(seq, kmer)
		f_prob = probabilities(seq, fc, kmer)
		# consider the evluation of the k.
		chaos_k = chaos_game_representation(f_prob, kmer)
		img = chaos_k

		# consider the efficiency here.
		# 依次比对batch中的label是否和训练的phage一一对应，如果一一对应说明是正样本。
		# 这里没有对负样本做其他处理，而是取出的每个phage都有host，因此需要根据y_train/y_val的host判断是1还是0
		for l in l2fa.keys():
			if l == label:
				labels.append(1)
				#print("Hit label:", l)
			else:
				labels.append(0)

			images.append(img)
			hosts.append(l2fa[l])

	return np.array(images), np.array(hosts), np.array(labels)


# standard approach of loading data for valdiation and testing.
def my_collate_fn2(batch, kmer):

	images, labels, phage_name_list = [],[],[]
	for name, seq, label in batch:
		phage_name = name
		seq = seq
		labels.append(label)

		# FCGR
		fc = count_kmers(seq, kmer)
		f_prob = probabilities(seq, fc, kmer)
		# consider the evluation of the k.
		chaos_k = chaos_game_representation(f_prob, kmer)
		img = chaos_k
		images.append(img)
		phage_name_list.append(phage_name)

	return np.array(images), np.array(labels), phage_name_list

'''
file_name 对应于 phage_train_file.fasta文件里面是噬菌体的DNA
label_file 对应于spiece_file所有的宿主
host_file 对应于XXX.csv其中每一条是宿主的name，与file_name中每一条噬菌体一一对应
'''
class fasta_dataset(Dataset):
	def __init__(self, file_name, label_file, host_file):
		# wgs表示当前Fasta这个文件，可以通过k-v获取name和DNA序列
		wgs = Fasta(file_name)
		self.name = [] # file文件中每一个phage的name
		self.seq = [] # file文件中每一个phage的DNA序列
		# 输入host文件，获得spieve_dic: {'name': 'label'}
		# label2species:host ['name']
		self.s2l_dic, self.l2s = get_label_map(label_file)

		# sequence process an put it in queue
		for pn in wgs.keys():
			self.name.append(pn)
			self.seq.append(wgs[pn][:].seq)
		# 为噬菌体self.name的host打标签按顺序.[167, 109, 18, 81, 60, 113, 80, 196, 76]
		self.label = load_host_label(host_file, self.s2l_dic)	

	def __len__(self):
		return	len(self.name)

	# 根据索引返回数据和对应的标签
	def __getitem__(self, idx):
		if(len(self.label) == 0):
			return self.name[idx], self.seq[idx], []
		# phage name, phage DNA, host id
		return self.name[idx], self.seq[idx], self.label[idx]

	# 获得spieve_dic: {'name': 'label'}
	def get_s2l_dic(self):
		return self.s2l_dic

	# label2species:host ['name']
	def get_l2s_dic(self):
		return self.l2s

class strain_fasta_dataset(Dataset):
	def __init__(self, strainPhage, strainPhageDNA, strainHost, strainHostDNA, strainLabel):
		self.strainPhage = strainPhage
		self.strainPhageDNA = strainPhageDNA
		self.strainHost = strainHost
		self.strainHostDNA = strainHostDNA
		self.strainLabel = strainLabel

	def __len__(self):
		return	len(self.strainPhage)

	# 根据索引返回数据和对应的标签
	def __getitem__(self, idx):
		if(len(self.strainLabel) == 0):
			return self.strainPhage[idx], self.strainPhageDNA[idx], []
		# phage name, phage DNA, host id
		return self.strainPhageDNA[idx], self.strainHostDNA[idx], self.strainLabel[idx]


def my_collate_strain(batch):

	# images表示每个phage的FCGR表示
	# hosts表示可能发生侵染关系的host FGCR
	# label表示image-host之间是否有相互作用
	strainPhageFCGR, strainHostFCGR, strainLabel = [],[],[]
	# 对于每批32个phage，对于训练的187的host都要测一下
	# 这样的话每个批次的phage-host-label  都变成了256*39=9984  （除了最后一个批次）
	for phage, host, label in batch:
		strainPhageFCGR.append(phage)
		strainHostFCGR.append(host)
		strainLabel.append(label)
		# 变成numpy即可
	return np.array(strainPhageFCGR), np.array(strainHostFCGR), np.array(strainLabel)


def saveLoss(epoch_loss_list, filename):
	path = 'results/'+filename
	str_result = ''
	for element in epoch_loss_list:
		str_result += str(element) + "\n"

	with open(path, mode='w', encoding="utf-8") as f:
		f.write(str_result)
	f.close()