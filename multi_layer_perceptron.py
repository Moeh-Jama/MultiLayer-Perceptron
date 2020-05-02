from simplyMLP import Matrix
import math

class MultiLayerPerceptron():

	def __init__(self, n_inputs, n_hidden, n_outputs):
		self.number_inputs = n_inputs
		self.number_hidden_units = n_hidden
		self.number_ouputs = n_outputs
		self.weights_of_ll = Matrix(n_hidden, n_inputs) # weights of lower layers
		self.weights_of_ul = Matrix(n_outputs, n_inputs) # weights of upper layers
		self.weights_of_ll.randomize()
		self.weights_of_ul.randomize()


		self.bias_hidden = Matrix(n_hidden,1)
		self.bias_output = Matrix(n_outputs,1)
		self.bias_hidden.randomize()
		self.bias_output.randomize()

		self.weight_changes_wll = 5*[[1,2,3]] # weight changes to be applied to weights_of_ll
		self.weight_changes_wul = 5*[[1,2,3]] # weight changes to be applied to weights_of_ul

		self.activations_ll = [] # activations for the lower layers
		self.activations_ul = [] # activations for the upper layers

		self.hidden_neuron_values = [] # values of the hidden neurons. for the computation of  weight_changes_wul
		self.final_outputs = [] # where the outputs are stored

	def sigmoid(self, val):
		return 1/(1+math.exp(-val))


	def randomise(self):
		# initialise wll and wul to small random values.
		# set wcll and wcul to all zeros.

		self.set_weightChanges_to_zero(self.weight_changes_wll)
		self.set_weightChanges_to_zero(self.weight_changes_wul)
		pass


	def activation_function(self, value):
		return value # do nothing

	def set_weightChanges_to_zero(self, wcl):
		for i in range(len(wcl)):
			for j in range(len(wcl[i])):
				wcl[i][j] = 0


	def forward(self, X):

		# take current value. Multiply it by the weight, send it to the input
		temp = Matrix(2,2)
		temp.matrix = X

		X = temp.arrayToMatrix(X)
		hidden = self.weights_of_ll.product(X)
		hidden = hidden + self.bias_hidden
		hidden = temp.arrayToMatrix(hidden)
		# activation 
		for i in range(hidden.rows):
			for j in range(hidden.cols):
				hidden.matrix[i][j] = self.sigmoid(hidden.matrix[i][j])

		output = self.weights_of_ul.product(hidden)
		# output = temp.arrayToMatrix(output)
		# print(self.bias_output.matrix)
		output = output + self.bias_output.matrix[0][0]
		output = temp.arrayToMatrix(output)
		# print(output)
		for i in range(output.rows):
			for j in range(output.cols):
				output.matrix[i][j] = self.sigmoid(output.matrix[i][j])
		normal_flat_array = []
		for i in range(output.rows):
			for j in range(output.cols):
				normal_flat_array.append(output.matrix[i][j])

		return output

	def backwards(self, target_t):
		# output-error, gradient.. etc.
		pass

	def updateWeights(self, learning_rate):
		self.weight_changes_wll = 0
		self.weight_changes_wul = 0
		pass


mlp = MultiLayerPerceptron(2,2,1)
inp = [[1],[0]]

output = mlp.forward(inp)
print(output.matrix)

