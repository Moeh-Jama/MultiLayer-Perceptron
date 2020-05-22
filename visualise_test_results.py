import matplotlib.pyplot as plt


path = "tests/xor_epoch_250_lr_35.txt"
error_rates = []
with open(path,'r') as f:
	lines = f.readlines()
	for line in lines:
		# print(line)
		line = line.strip()
		
		if line.split(" ")[0].strip()=="Error":
			score = float(line.split("is ")[-1])
			score = round(score,4)
			print(score)
			error_rates.append(score)


print(score)
plt.plot(error_rates)
plt.ylabel('some numbers')
plt.show()