import math
import numpy as np
import random
class MultiLayerPerceptron():

	def __init__(self, n_inputs, n_hidden, n_outputs, loss_function, bias=1, batch=1):
		self.number_inputs = n_inputs
		self.number_hidden_units = n_hidden
		self.number_ouputs = n_outputs


		self.output_steps = 50

		self.batch = batch
		
		self.weights_of_ll = self.initialise_weights(self.number_inputs+bias, self.number_hidden_units) # weights of lower layers
		self.weights_of_ul = self.initialise_weights(self.number_hidden_units, self.number_ouputs) # weights of upper layers

		self.inputs = [1]*(self.number_inputs + 1)

		#Set the activation and derivative functions. Send the functions in and store them in the MLP Object
		self.activation_function = np.tanh
		self.derivative_activation_function = self.derivative_tanh

		self.loss_function = loss_function

		self.hidden_neuron_values = [1]*self.number_hidden_units # values of the hidden neurons. for the computation of  weight_changes_wul
		self.outputs = [1]*self.number_ouputs # where the outputs are stored

		self.learning_rate = 0.1
		self.isUpdated = False
		self.weight_changes_wll =0 # weight changes to be applied to weights_of_ll
		self.weight_changes_wul = 0 # weight changes to be applied to weights_of_ul
		# self.activations_ll = [] # activations for the lower layers
		# self.activations_ul = [] # activations for the upper layers

		self.display = True
	

	def initialise_weights(self, rows, cols):
		weights = []
		for i in range(rows):
			vector = []
			for j in range(cols):
				vector.append(random.uniform(-0.2,0.2))
			weights.append(vector)
		return np.array(weights, dtype=float)

	def sigmoid(self, val):
		return 1/(1+math.exp(-val))

	def derivative_sigmoid(self, y):
		return y * (1-y)
	
	def softmax(self, v):
		e = np.exp(v - v.max())
		return e/np.sum(e)

	def derivative_softmax(self, output):
		return self.softmax(output) * (1 - self.softmax(output))


	def derivative_tanh(self, v):
		return 1 - v**2

	def randomise(self):
		# initialise wll and wul to small random values.
		# set wcll and wcul to all zeros.

		self.set_weightChanges_to_zero(self.weight_changes_wll)
		self.set_weightChanges_to_zero(self.weight_changes_wul)
		


	# def activation_function(self, value):
	# 	return value # do nothing
	

	# def derivative_activation(self, value):
	# 	return value

	def set_weightChanges_to_zero(self, wcl):
		for i in range(len(wcl)):
			for j in range(len(wcl[i])):
				wcl[i][j] = 0


	def forward(self, X):
		self.inputs[:-1] = X

		self.hidden_neuron_values = self.activation_function(np.dot(self.inputs, self.weights_of_ll))

		#The output production depends on the type of process we are running. If we are doing classificaiton 
		# i.e. non-continuous single outputs, we must use an appropriate function like 
		if self.number_ouputs >1:
			self.outputs = self.softmax(np.dot(self.hidden_neuron_values, self.weights_of_ul))
		else:
			self.outputs = self.activation_function(np.dot(self.hidden_neuron_values, self.weights_of_ul))
		
		return self.outputs


	def backwards(self, target_t):
		# output-error, gradient.. etc.
		
		error = target_t - self.outputs

		# calculation of the deltas of the output layer
		delta_functions = None
		if self.number_ouputs > 1:
			outer_delta_functions = error * self.derivative_softmax(self.outputs)
		else:
			outer_delta_functions = error * self.derivative_activation_function(self.outputs)
		
		hidden_delta_functions = np.dot(outer_delta_functions, self.weights_of_ul.T) * self.derivative_activation_function(self.hidden_neuron_values)

		if self.isUpdated:
			self.weight_changes_wll += np.dot(np.atleast_2d(self.inputs).T, np.atleast_2d(hidden_delta_functions))
			self.weight_changes_wul += np.dot(np.atleast_2d(self.hidden_neuron_values).T, np.atleast_2d(outer_delta_functions))
		else:
			self.weight_changes_wll = np.dot(np.atleast_2d(self.inputs).T, np.atleast_2d(hidden_delta_functions))
			self.weight_changes_wul = np.dot(np.atleast_2d(self.hidden_neuron_values).T, np.atleast_2d(outer_delta_functions))


		# update the weights after this function is called, *sometimes...





	def train(self, examples,labels, iteration, learning_rate=0.1):
		error_rates = []
		for epoch in range(iteration):
			error = 0

			for i in range(len(examples)):

				example = examples[i]
				label = labels[i]
				output = self.forward(example)
				
				error += self.loss_function(output, label)

				self.backwards(label)
				if i%self.batch==0:
					# print('batch job - update weights. Batches every {}'.format(self.batch))
					self.updateWeights(learning_rate)
			if epoch%self.output_steps==0 and self.display:
				print("Error at epoch {} is {}".format(epoch, error/len(examples)))
			error_rates.append(error/len(examples))
		return error_rates



	def updateWeights(self, learning_rate):
		self.weights_of_ll += learning_rate*self.weight_changes_wll
		self.weights_of_ul += learning_rate*self.weight_changes_wul
		self.weight_changes_wll = 0
		self.weight_changes_wul = 0
		self.isUpdated = not self.isUpdated