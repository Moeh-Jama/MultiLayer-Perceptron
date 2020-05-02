from multi_layer_perceptron import MultiLayerPerceptron


class NeuralNetwork():

	def __init__(self):
		pass

	def assign_data_source(self, filepath):
		pass

	def write_output(self, filepath):
		pass

	def train(self, epochs, examples):

		for i in range(epochs):
			error = 0
			for j in range(len(examples)):

				error+= 0 # NN.backwards(examples[j].inputs)
				pass
			print('Error at epoch {} is {}'.format(i,error))
		pass

nn = NeuralNetwork()
nn.train(5,[0]*5)