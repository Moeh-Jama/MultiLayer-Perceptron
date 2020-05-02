
import random
class Matrix():

	def __init__(self, rows, cols):
		self.rows = rows
		self.cols = cols
		self.matrix = []
		for i in range(rows):
			row = []
			for j in range(cols):
				row.append(0)
			self.matrix.append(row)

	def arrayToMatrix(self, array):
		if type(array[0])!=list:
			m = Matrix(1,len(array))
			m.matrix = array
			return m
		else:
			m =Matrix(len(array), len(array[0]))
			m.matrix = array
			return m


	def randomize(self):
		for i in range(self.rows):
			for j in range(self.cols):
				self.matrix[i][j] = random.randint(-1,1)

	def __mul__(self, other):
		if type(other) == list:
			pass
		elif type(other) == float or type(other)==int:
			for i in range(self.rows):
				for j in range(self.cols):
					self.matrix[i][j] *= other
		return self.matrix


	def __add__(self, other):
		# print(type(other), type(other)==Matrix)
		if type(other)==list:
			for i in range(self.rows):
				for j in range(self.cols):
					self.matrix[i][j] += other[i][j]
		elif type(other) == Matrix:
			# other = self.matrix
			matching_size_matrices =  (other.cols==self.cols and self.rows==other.rows)
			# print(len(other[0]),self.cols, self.rows, len(other))
			if not matching_size_matrices:
				raise Exception("The given matrix does not match the size of the matrix it is being added with!\n other matrix is size {}*{} and current matrix is {}*{}".format(other.rows,other.cols,self.rows,self.cols))
			for i in range(self.rows):
				for j in range(self.cols):
					self.matrix[i][j] += other.matrix[i][j]
		elif type(other)==float or type(other)==int:
			for i in range(self.rows):
				for j in range(self.cols):
					self.matrix[i][j] += other
		return self.matrix



	def product(self, other):
		if type(other)!= Matrix:
			raise Exception("Matrix product cannot be returned with {} object type used".format(type(other)))
		# print(other.rows, self.cols)
		if other.rows != self.cols:
			raise Exception("Matrix product cannot be gotten with non-well defined sizes of Matrix sizes ({},{}) * ({},{})".format(self.rows,self.cols, other.rows,other.cols))

		product = Matrix(self.rows, other.cols)

		for i in range(product.rows):
			for j in range(product.cols):
				val = 0
				for k in range(self.cols):
					val+= self.matrix[i][k]*other.matrix[k][j]
				product.matrix[i][j] = val
		return product

	def transpose(self):

		transposed_matrix = Matrix(self.cols,self.rows)

	def __str__(self):
		return "{}x{}".format(self.rows,self.cols)

# matrix = Matrix(2,3)
# matrix.randomize()
# m = Matrix(3,2)
# m.randomize()
# print(matrix.matrix)
# print(m.matrix)

# matrix.matrix = [[6,7,0],[7,2,6]]
# m.matrix = [[5,3],[1,1],[5,1]]


# result = matrix.product(m)
# print(result.matrix)