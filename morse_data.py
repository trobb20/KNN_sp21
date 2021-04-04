#morse code getter
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def import_data(f):
	df = pd.read_csv(f)
	timeCol = df.columns[0]
	lightCol = df.columns[1]

	t = df[timeCol].to_numpy()
	light = df[lightCol].to_numpy()

	maximum = np.max(light)
	minimum = np.min(light)

	light = 2*((light-minimum)/(maximum-minimum))-1

	return np.array([t,light])

def find_zeros(y):
	zeros = np.empty(1)
	for i in range(1,y.shape[0]):
		y1 = y[i]
		y0 = y[i-1]

		if y0>=0 and y1<=0:
			zeros = np.append(zeros,i)
		elif y0<=0 and y1>=0:
			zeros = np.append(zeros,i)

	return zeros

def deltas(zeros):
	deltas = np.zeros(zeros.shape)

	for i in range(1,zeros.shape[0]):
		delta = zeros[i]-zeros[i-1]
		deltas[i-1]=delta

	return deltas

def produce_data(deltas):
	classes = np.random.randint(0,high=3,size=deltas.shape[0]).astype('str')
	data = np.array([deltas.T,classes], dtype=object).T

	return data


# t,light = import_data('morse.csv')
# plt.figure()
# plt.plot(t,light)
# plt.show()

# zeros = find_zeros(light)
# deltas = deltas(zeros)

# plt.figure()
# plt.plot(deltas,np.zeros(deltas.shape),'ro')
# plt.show()

# print(produce_data(deltas))