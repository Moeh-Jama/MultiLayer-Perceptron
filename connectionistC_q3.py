
import random
import numpy as np


def getVector_List(num_vectors=200, vector_size=4):
	# GENERATE 200 (4*1) vectors
	vector_list = []
	output_list = []
	for i in range(num_vectors):
		vector = []
		for j in range(vector_size):
			vector.append(random.uniform(-1,1))
		# print(vector)
		combination = vector[0]-vector[1] + vector[2]-vector[3]
		y = np.sin(combination)
		vector_list.append(vector)
		output_list.append(y)
	return vector_list,output_list

