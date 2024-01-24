import os
from pyfaidx import Fasta
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fasta2CGR import probabilities, count_kmers, chaos_game_representation


def getStrainData(kmer) :
    # 读取文件
    strainData = pd.read_excel(io='./data/Pheotype_20210913.xlsx')
    # 空白换成0
    strainData = strainData.replace(np.nan, 0)
    # 获取hostList
    hostList = [element for element in strainData.columns][1:]
    # 获得phageList
    phageList = list(strainData['ID'])

    # 将104*125变numpy，将综合评分化成1/0
    PBIs = np.delete(np.array(strainData), 0, 1)
    PBIs[PBIs<=1.5]=0
    PBIs[PBIs>1.5]=1

    # 删除全是0的五列数据，host:KP4829,KP5119,KP9778,KP9822,KP9932  host变为120条，phage仍为104
    mask = (PBIs == 0).all(0)
    column_indices = np.where(mask)[0]
    PBIs = PBIs[:,~mask]
    hostList.remove('KP4829')
    hostList.remove('KP5119')
    hostList.remove('KP9778')
    hostList.remove('KP9822')
    hostList.remove('KP9932')

    # 获得120host和104phage的sequence
    host_fasta_path = './data/bac_fasta_2021/'
    phage_fasta_path = './data/phage_fasta_2021/'
    host_fast_pathList = [host_fasta_path+element+'.fasta' for element in hostList]
    phage_fast_pathList = [phage_fasta_path+element+'.fasta' for element in phageList]

    strainHostDNA_120 = []
    for file_name in host_fast_pathList:
        wgs = Fasta(file_name)
        for bn in wgs.keys():
            # 计算给定seq的kmer的频率
            seqence = wgs[bn][:].seq
            fc = count_kmers(seqence, kmer)
            # 计算给定seq和fc的kmer概率   即每个k-mer的频率换成k-mer概率
            f_prob = probabilities(seqence, fc, kmer)
            chaos_k = chaos_game_representation(f_prob, kmer)
            strainHostDNA_120.append(chaos_k)
            break

    strainPhageDNA_104 = []
    for file_name in phage_fast_pathList:
        wgs = Fasta(file_name)
        for bn in wgs.keys():
            # 计算给定seq的kmer的频率
            seqence = wgs[bn][:].seq
            fc = count_kmers(seqence, kmer)
            # 计算给定seq和fc的kmer概率   即每个k-mer的频率换成k-mer概率
            f_prob = probabilities(seqence, fc, kmer)
            chaos_k = chaos_game_representation(f_prob, kmer)
            strainPhageDNA_104.append(chaos_k)
            break


    # 一共12480条数据
    strainPhage, strainPhageDNA, strainHost, strainHostDNA, strainLabel = [], [], [], [], []
    for i in range(PBIs.shape[0]):
        phages = [phageList[i] for _ in range(PBIs.shape[1])]
        phagesDNA = [strainPhageDNA_104[i] for _ in range(PBIs.shape[1])]
        strainPhage.extend(phages)
        strainPhageDNA.extend(phagesDNA)
        strainHost.extend(hostList)
        strainHostDNA.extend(strainHostDNA_120)
        strainLabel.extend(PBIs[i][:])
    return strainPhage, strainPhageDNA, strainHost, strainHostDNA, strainLabel