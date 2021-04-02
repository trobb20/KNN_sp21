#KNN Example
#Teddy Robbins ES2TA sp 2021
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# Our dataset is a numpy array with vectors corresponding
# to datapoints. Each datapoint is assigned a 'classification'
# which is tacked on to the end of the vector. 
data = np.array([[1,1,'a'],
				[1.1,1.1,'down'],
				[1.2,1.2,'up'],
				[0.9,0.8, 'up'],
				[0.1,0.1, 'down'],
				[0.2,0.2, 'x'],
				[-0.1,-0.1,'up'],
				[0,0,'down'],
				[0.6,0.6,'x']])

def get_vector(data,ind):
	# Given the dataset, and an index for a vector,
	# return just the vector without the associated classification
	vector = data[ind][:-1].astype(np.float)
	return vector

def get_distance(vector1, vector2):
	# Compute the distance between two vectors in n-dimensions
	dist = np.linalg.norm(vector1-vector2)
	return dist

def get_neighbors(data,ind):
	# For one datapoint in data with index ind, compile
	# a sorted list of distances to other datapoints. 
	# Format for output array is neighbors = [distances, indecies]

	#Initialize neighbors
	neighbors = np.zeros((data.shape[0],2))

	#Compare our datapoint to every other datapoint
	for i in range(0,data.shape[0]):
		#Get the vectors from the datapoints
		classify_vector = get_vector(data,ind)
		compare_vector = get_vector(data,i)
		#Get their distance
		dist = get_distance(classify_vector,compare_vector)
		#Add it to the array
		neighbors[i] = np.array([dist,i])

	#Sort by first column and elimnate first entry (our datapoint is closest to itself.)
	neighbors = neighbors[neighbors[:,0].argsort()]
	neighbors = np.delete(neighbors,0,0)

	return(neighbors)

def classify(data, neighbors, ind, K):
	# Get a list of neighbors, and a datapoint with index ind in data.
	# Given K as the number of nearest neighbors, get those
	# neighbors from the neighbors array, and find the most common
	# classification among them. Return this classification.

	#Setup classes array
	classes = np.empty(K,dtype=object)

	#Get K nearest neighbors
	for i in range(0,K):
		#Look up their index from the neighbors array
		lookup_ind = int(neighbors[i,1])
		#Get their class from the data array
		current_class = data[lookup_ind,-1]
		#Assign that class to our running list
		classes[i] = current_class

	#Use scipy.stats to find the most common classification in classes
	classification = st.mode(classes).mode[0]

	return classification

def KNN_main(data,K,verbose=False):
	# Main function. Loops through each datapoint in data
	# and re-assigns its class based on its K nearest neighbors.
	# This function edits the data array, so make sure to make a copy
	# before running it if you want the old data. This function depends
	# on all other functions.

	#Loop through data
	for i in range(data.shape[0]): 
		#For each point, reassign classification using previous functions
		neighbors = get_neighbors(data,i)
		classification = classify(data,neighbors,i,K)

		if verbose:
			print('-----------------------------------------------')
			print('Looking at datapoint #%d, = %s'%(i,str(data[i])))
			print('Nearest neighbors list is: \n %s'%(str(neighbors)))
			print('Found common classification: %s'%(str(classification)))
			print('Finished with datapoint. Proceeding...')

		data[i,-1]=classification

	return

######## Testing ##########
old_data=np.copy(data)
K=3
KNN_main(data,K,verbose=True)

#Visualize datasets
fig1 = plt.figure()
plt.title('old data')
for i in range(old_data.shape[0]):
	vec = get_vector(old_data,i)
	if old_data[i,-1]=='up':
		plt.plot(vec[0],vec[1],'ro')
	elif old_data[i,-1]=='down':
		plt.plot(vec[0],vec[1],'bo')
	else:
		plt.plot(vec[0],vec[1],'ko')

fig2 = plt.figure()
plt.title('new data')
for i in range(data.shape[0]):
	vec = get_vector(data,i)
	if data[i,-1]=='up':
		plt.plot(vec[0],vec[1],'ro')
	elif data[i,-1]=='down':
		plt.plot(vec[0],vec[1],'bo')
	else:
		plt.plot(vec[0],vec[1],'ko')

plt.show()