# implemenation of Chaos Game Represenation of a genetic sequence.
# code reference: https://towardsdatascience.com/chaos-game-representation-of-a-genetic-sequence-4681f1a67e14

import collections
import math
from collections import defaultdict

# def empty_dict():
# 	"""
# 	None type return vessel for defaultdict
# 	:return:
# 	"""
# 	return None

#计算kmer
'''
计算kmer
in: DNA序列
out: defaultdict {'6-mer':次数}
'''
def count_kmers(sequence, k):
	d = collections.defaultdict(int)
	for i in range(len(sequence)-(k-1)):
		d[sequence[i:i+k]] += 1

	for key in list(d.keys()):
		if "N" in key:  # 正在分析的序列还包含一个 N，并且这个 N 可以是 A、C、G 或 T 中的任何一个，具体取决于概率，删除它
			del d[key]
	return d


'''
计算k-mer出现的频率
in: data为序列的长度，kmer dict
out:k-mer:frequency
'''
def probabilities(data, kmer_count, k):
	probabilities = collections.defaultdict(float)
	N = len(data)
	for key, value in kmer_count.items():
		probabilities[key] = float(value) / (N - k + 1)
	return probabilities

'''
CGR
chaos 64X64
'''
def chaos_game_representation(probabilities, k):
	array_size = int(math.sqrt(4**k))
	chaos = []
	for i in range(array_size):
		chaos.append([0]*array_size)

	maxx, maxy = array_size, array_size
	posx, posy = 1, 1

	for key, value in probabilities.items():
		for char in key:
			if char == "T" or char == "t":
				posx += maxx / 2
			elif char == "C" or char == "c":
				posy += maxy / 2
			elif char == "G" or char == "g":
				posx += maxx / 2
				posy += maxy / 2
			maxx = maxx / 2
			maxy /= 2

		chaos[int(posy)-1][int(posx)-1] = value
		maxx = array_size
		maxy = array_size
		posx = 1
		posy = 1

	return chaos


# def cgr_positions(seq):
#
# 	# CGR_CENTER = (0.5, 0.5)
# 	CGR_X_MAX, CGR_Y_MAX = 1, 1
# 	CGR_X_MIN, CGR_Y_MIN = 0, 0
#
# 	CGR_A = (CGR_X_MIN, CGR_Y_MIN)
# 	CGR_T = (CGR_X_MAX, CGR_Y_MIN)
# 	CGR_G = (CGR_X_MAX, CGR_Y_MAX)
# 	CGR_C = (CGR_X_MIN, CGR_Y_MAX)
# 	CGR_CENTER = ((CGR_X_MAX - CGR_Y_MIN) / 2, (CGR_Y_MAX - CGR_Y_MIN) / 2)
#
# 	CGR_DICT = defaultdict(
# 	empty_dict,
# 	[
# 		('A', CGR_A),  # Adenine
# 		('T', CGR_T),  # Thymine
# 		('G', CGR_G),  # Guanine
# 		('C', CGR_C),  # Cytosine
# 		('U', CGR_T),  # Uracil demethylated form of thymine
# 		('a', CGR_A),  # Adenine
# 		('t', CGR_T),  # Thymine
# 		('g', CGR_G),  # Guanine
# 		('c', CGR_C),  # Cytosine
# 		('u', CGR_T)  # Uracil/Thymine
# 		]
# 	)
#
# 	cgr = []
# 	cgr_marker = CGR_CENTER[:
# 		]    # The center of square which serves as first marker
# 	for s in seq:
# 		cgr_corner = CGR_DICT[s]
# 		if cgr_corner:
# 			cgr_marker = (
# 				(cgr_corner[0] + cgr_marker[0]) / 2,
# 				(cgr_corner[1] + cgr_marker[1]) / 2
# 			)
# 			cgr.append([s, cgr_marker])
# 		else:
# 			#sys.stderr.write("Bad Nucleotide: " + s + " \n")
# 			pass
#
# 	return cgr

