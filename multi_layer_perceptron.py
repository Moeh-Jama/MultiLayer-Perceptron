

class MultiLayerPerceptron():

	def __init__(self, n_inputs, n_hidden, n_outputs):
		self.number_inputs = n_inputs
		self.number_hidden_units = n_hidden
		self.number_ouputs = n_outputs
		self.weights_of_ll = None # weights of lower layers
		self.weights_of_ul = None # weights of upper layers

		self.weight_changes_wll = 5*[[1,2,3]] # weight changes to be applied to weights_of_ll
		self.weight_changes_wul = 5*[[1,2,3]] # weight changes to be applied to weights_of_ul

		self.activations_ll = [] # activations for the lower layers
		self.activations_ul = [] # activations for the upper layers

		self.hidden_neuron_values = [] # values of the hidden neurons. for the computation of  weight_changes_wul
		self.final_outputs = [] # where the outputs are stored


	def randomise(self):
		# initialise wll and wul to small random values.
		# set wcll and wcul to all zeros.

		self.set_weightChanges_to_zero(self.weight_changes_wll)
		self.set_weightChanges_to_zero(self.weight_changes_wul)
		pass

	def set_weightChanges_to_zero(self, wcl):
		for i in range(len(wcl)):
			for j in range(len(wcl[i])):
				wcl[i][j] = 0


	def forward(self, vector):
		pass

	def backwards(self, target_t):
		pass

	def updateWeights(self, learning_rate):
		self.weight_changes_wll = 0
		self.weight_changes_wul = 0
		pass
