import random
import connectionistC_q3 as q3
import numpy as np

from multi_layer_perceptron import MultiLayerPerceptron




def cross_entropy(output, target):
	mul = np.log(1-output) * np.log(output) - (1-target)  * -(target)
	mul  = np.nan_to_num(mul)
	loss = np.sum(mul)
	return loss

def squared_error(output, target):
	return 0.5* ((target - output)**2).sum()

class Example():

	def __init__(self, input_data, output_data):
		self.input = input_data
		self.output = output_data



def learn_XOR():
	
	vectors = [[0,0],[0,1],[1,0],[1,1]]
	outputs = [0,1,1,0]
	example_list = []
	for i in range(len(vectors)):
		example = Example(vectors[i],outputs[i])
		example_list.append(example)
	
	#Shuffle??
	random.shuffle(example_list)
	train_v = []
	train_o = []
	for example in example_list:
		train_v.append(example.input)
		train_o.append(example.output)
	inputs = np.array((train_v), dtype=float)
	labels  = np.array((train_o), dtype=float)

	nn = MultiLayerPerceptron(2,3,1,squared_error)
	nn.train(inputs,labels, 150,learning_rate=0.45)
	random.shuffle(example_list)

	for example in example_list:
		X = example.input
		Y = example.output
		output = nn.forward(X)
		print(X,output)

		
	


def learn_Sin():
	nn = MultiLayerPerceptron(4,5,1,squared_error)

	vectors,outputs = q3.getVector_List()

	#train data
	train_v = []
	train_o = []

	#test data
	test_v = []
	test_o = []
	example_list = []
	for i in range(len(vectors)):
		example = Example(vectors[i],outputs[i])
		example_list.append(example)
	random.shuffle(example_list)
	lim = int(len(example_list) * (2/3))
	for example in example_list[:lim]:
		train_v.append(example.input)
		train_o.append(example.output)
	
	inputs = np.array((train_v), dtype=float)
	labels  = np.array((train_o), dtype=float)
	nn.train(inputs, labels, 150,learning_rate=0.1)


def readDigitRecognition():
    c = 0
    label_mapping = {}
    map_label = {}

    file_name = "letter-recognition.data"
    path = "dataset/"+file_name
    vectors = []
    labels = []
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split(",")
            v = line[1:]
            l = line[0]

            # print(line)
            if l not in label_mapping:
                label_mapping[l] = c
                map_label[c] = l
                l = c
                c+=1
            else:
                l = label_mapping[l]
            letters = [0]*26
            letters[l] = 1
            v = [int(v_i) for v_i in v]
            vectors.append(v)
            labels.append(letters)
    return vectors, labels, label_mapping, map_label




def learn_digit_recognition():
	nn = MultiLayerPerceptron(16,17,26,cross_entropy, batch=1)

	# vectors,outputs = q3.getVector_List()
	vectors, outputs, label_mapping, map_label = readDigitRecognition()
	#train data
	train_v = []
	train_o = []

	#test data
	test_v = []
	test_o = []
	example_list = []
	for i in range(len(vectors)):
		example = Example(vectors[i],outputs[i])
		example_list.append(example)
	random.shuffle(example_list)
	lim = int(len(example_list) * (2/3))
	for example in example_list[:lim]:
		train_v.append(example.input)
		train_o.append(example.output)
	
	inputs = np.array((train_v), dtype=float)
	labels  = np.array((train_o), dtype=float)
	nn.train(inputs, labels, 50,learning_rate=0.1)

	for example in example_list[lim:]:
		test_v.append(example.input)
		test_o.append(example.output)
	
	correct = 0
	for i in range(len(test_v)):
		example_in = test_v[i]
		example_out = test_o[i]

		out = list(nn.forward(example_in))
		out = out.index(max(out))
		example_out = example_out.index(max(example_out))

		if out == example_out:
			correct+=1
	acc = (correct/len(test_v))*100
	print("Accuracy is {}%".format(acc))



def main():
	learn_XOR()
	# learn_Sin()
	# learn_digit_recognition()







if __name__ == '__main__':
	# method for adding arguements
	# parser = ArgumentParser()
	# parser.add_arguement("")
	main()
