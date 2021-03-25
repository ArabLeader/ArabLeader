# Tom Rishel
# University of Southern Mississippi
# March 2021
# This program will implement a neural network to classify unicode documents in
# Arabic into terrorist-related and non-terrorist-related categories. We 
# previously harvested ~1.2 million tweets from 14 middle eastern countries and
# scored them using LSA to calculate a cosine similarity to a query document. We 
# used Otsu's method to find a threshold and took all of the documents above the 
# threshold as the positivelly correlated set. We used an equal number of documents
# with the lowest cosine similarities as the negative set. We divide both sets
# into training and test sets with 80% for training and 20% for testing.

import sys
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os, glob, glob2

from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split

import matplotlib as mp
import matplotlib.pyplot as plt

from pathlib import Path


# custom one hot encoding function
def OneHotEncode1D(data):
	counter = 0
	oneCounter = 0
	outer_dim = max(data) + 1
	out = []
	for element in data:
		inner = []
		if element == 1:
			#print("found a 1 at counter " + str(counter))
			oneCounter += 1
		for i in range(outer_dim):
			if element == i:
				inner.append(1)
			else:
				inner.append(0)
		out.append(inner)
		counter += 1
		#print(element)
	print("Out of " + str(counter) + " elements, " + str(oneCounter) + " were 1's. " + str((oneCounter/counter)*100) + "%")
	return out


# This function takes two lists. Each entry in the first list is a string of 
# text. The second list is the known classification value (0 for non-terror-related, 
# 1 for terror-related) of each item in the first list. It creates a neural network 
# model and trains the model using the provided parameters.
def listToModel(training_text, training_target):

	# set up the model_folder variable to the folder to store the trained model
	model_folder = Path("NLP_NN/")
	
	# use tf-idf weighting to select the most significant terms from the data
	# be sure to change the architecture of the network in the configModel
	# function to match the number of input vectors
	#input_length = 5000
	#input_length = 2500
	#input_length = 1000
	#input_length = 500
	#input_length = 250
	#input_length = 100
	input_length = 50
	#input_length = 30

	# this line applies tf-idf values from the document set and excludes those
	# terms with a document frequency greater than max_df
	vectorizer = text.TfidfVectorizer(max_features=input_length, max_df=0.70)

	# use the scikit learn function to split the dataset randomly into training and test sets
	# 80% training 20% test
	train_text, test_text, train_class, test_class = train_test_split(training_text, training_target, test_size=0.20)#, random_state=42)

	# text_train = the training text
	text_train = vectorizer.fit_transform(train_text)
	text_train = torch.FloatTensor(text_train.toarray())

	print("text_train size: " + str(text_train.size()))
	
	# known_class_train = the known class of the training data. 
	# convert the training targets into a torch tensor
	known_class_train = torch.FloatTensor(np.array(OneHotEncode1D(train_class)))
	#known_class_train = torch.FloatTensor(np.array(train_class))

	# text_test = the test data
	text_test = vectorizer.transform(test_text)
	text_test = torch.FloatTensor(text_test.toarray())
	print("Shape of text_test:\t" + str(text_test.shape))

	# known_class_test = the known class of the test data 
	known_class_test = torch.FloatTensor(np.array(OneHotEncode1D(test_class)))
	#known_class_test = torch.FloatTensor(np.array(test_class))
	print("Shape of known_class_test:\t" + str(known_class_test.shape))

	# the output vectors should be the same length as the number of classes, in this case, 2
	output_length = 2

	num_epochs = 10000

	# create the NN model
	model = configModel(input_length, output_length)

	loss_fn = torch.nn.MSELoss(reduction='sum')
	learning_rate = 1e-4
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

	# set up learning rate decay
	lr_step_size = math.floor(num_epochs/40)
	lr_gamma = 0.85
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

	# set the model to training mode
	model.train()
	
	# calcuate the time required to train the model
	t0 = time.time()
	#t3 = str(time.time())

	#now = datetime.now()

	fname = timeStamped("loss_curve.csv")
	loss_curve = open(fname, "w")
	# if wanting to keep a csv file of validation loss values, uncomment this line and
	# the "write" line below
	#fname = timeStamped("validationLoss_curve.csv")
	#validationLoss_curve = open(fname, "w")

	#plot the loss curve
	fig1 = plt.figure()
	label = "Loss curve for " + str(num_epochs) + " training epochs"
	ax = plt.axes(label=label)
	plt.title(label)
	plt.xlabel("Number of Epochs")
	plt.ylabel("Loss")
	#ax.legend()

	# set up the file to save the incremental accuracy
	accuracyFileName = timeStamped("accuracy_" + str(num_epochs) + ".txt")
	accuracyOutFile = open(accuracyFileName, 'w')

	loss_list = []
	#validationLoss_list = []
	print("Running " + str(num_epochs) + " epochs")

	for t in range(num_epochs):
		output_prediction = model(text_train)
		loss = loss_fn(output_prediction, known_class_train) #known_class_train needs to be a 2d tensor to match size of output prediction
		loss_list.append(loss.item())
		#print(str(t) + "," + str(loss.item()))
		loss_curve.write(str(t) + "," + str(loss.item()) + "\n")
		ax.scatter(t, loss.item(), color="black", s=1)

		if t % 100 == 0:
			print('Epoch-{0} learning rate: {1} loss: {2}'.format(t, optimizer.param_groups[0]['lr'], loss.item()))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		# list of epochs at which to check accuracy
		accuracyChecks = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000]

		if t in accuracyChecks:
			currentAccuracy = testAccuracy(t, model, text_test, known_class_test)
			# print accuracy to screen
			print("\nTotal accuracy at epoch " + str(t) + ": " + str(currentAccuracy[0]) + "%")
			print("Positive accuracy at epoch " + str(t) + ": " + str(currentAccuracy[1]) + "%")
			# write accuracy to file
			accuracyOutFile.write("\n\nTotal accuracy at epoch " + str(t) + ": " + str(currentAccuracy[0]) + "%")
			accuracyOutFile.write("\nPositive accuracy at epoch " + str(t) + ": " + str(currentAccuracy[1]) + "%")
			accuracyOutFile.write("\nLoss at epoch " + str(t) + ": " + str(loss.item()))

	# ending time
	t1 = time.time()
	et = t1 - t0
	print("Total training time: " + str(et) + "\nNumber of features:" + str(input_length) + 
		  "\nNumber of epochs:" + str(num_epochs))

	# save the model to disk
	modelFileName = timeStamped(str(model_folder) + ".pth")
	torch.save(model, modelFileName) #str(model_folder) + "_" + str(num_epochs) + "_" + str(now) + ".pth")
	#torch.save({
	#			'epoch': num_epochs,
	#			'model_state_dict': model.state_dict(),
	#			'optimizer_state_dict': optimizer.state_dict(),
	#			'loss': loss
	#			}, str(model_folder) + "_" + str(num_epochs) + "_" + str(now) + ".pth")

	
	# calculate and plot the validation loss
	
	# set the model to evaluation mode
	#model.eval()
	#for t in range(num_epochs):
	#	output_prediction = model(text_test)
	#	validationLoss = loss_fn(output_prediction, known_class_test) #known_class_train needs to be a 2d tensor to match size of output prediction (right?)
	#	validationLoss_list.append(validationLoss.item())
	#	#print(str(t) + "," + str(validationLoss.item()))
	#	#validationLoss_curve.write(str(t) + "," + str(validationLoss.item()) + "\n")
	#	ax.scatter(t, validationLoss.item(), color="blue", s=1)


	# test the model and output the accuracy
	print("Starting final testing")
	correct = 0
	counter = 0
	positive_correct = 0
	positive_count = 0
	for item in text_test:
		output = model(item)
		#predicted = output.data.cpu().numpy()
		predicted = output.data.cpu().numpy().argmax()
		#print("predicted class: " + str(predicted))
		#print("actual class: " + str(known_class_test[counter].numpy()))
		check = known_class_test[counter].argmax()
		if predicted == known_class_test[counter].argmax():
			correct += 1
		if known_class_test[counter].argmax() == 1:
			positive_count += 1
			if predicted == known_class_test[counter].argmax():
				positive_correct += 1
		counter += 1

	# print accuracy to the screen
	print("Out of " + str(counter) + " articles the model predicted " + str(correct) +  
		  " correctly. " + str(round(correct/counter * 100, 7)) + "%")
	print("Out of " + str(positive_count) + " positive articles the model predicted " + 
	   str(positive_correct) + " correctly. " + str(round(positive_correct/positive_count * 100, 7)) + "%")
	# write accuracy to the file
	accuracyOutFile.write("\n\nOut of " + str(counter) + " articles the model predicted " + str(correct) +  
		  " correctly. " + str(round(correct/counter * 100, 7)) + "%")
	accuracyOutFile.write("\nOut of " + str(positive_count) + " positive articles the model predicted " + 
	   str(positive_correct) + " correctly. " + str(round(positive_correct/positive_count * 100, 7)) + "%")

	fig1.text(0.2, 0.4, "Total accuracy: " + str(round(correct/counter * 100, 3)) + "%" + 
		   "\nAccuracy of positives: " + str(round(positive_correct/positive_count * 100, 3)) + "%" +
		   "\nNumber of features: " + str(input_length) + 
		   "\nNumber of epochs: " + str(num_epochs) + 
		   "\nTime: " + str(round(et,3)) + 
		   "\nLearning rate base: " + str(learning_rate) + 
		   "\nLearning rate steps: step_size=" + str(lr_step_size) + ", gamma=" + str(lr_gamma), 
		   bbox=dict(facecolor = 'white', alpha=0.5), transform=ax.transAxes)
	# save the loss plot and close the output files
	fig1Filename = timeStamped("Loss" + str(num_epochs) + ".png")
	fig1.savefig(fig1Filename, bbox_inches="tight") #"Loss " + str(num_epochs) + "_" + str(now) + ".png", bbox_inches="tight")
	loss_curve.close()
	accuracyOutFile.close()

	#####################################################################################
	# run the twitter dataset against the model

	#dataset_text = filesToDocs("<path to data folder goes here>\\TKX\\twitter_screen_names_by_country", "terrorHits.csv")

	# set the model to evaluation mode
	#model.eval()
	# - or -
	#model.train()

	# dataset = the twitter data
	#dataset = vectorizer.transform(dataset_text)
	#dataset = torch.FloatTensor(dataset.toarray())
	#counter = 0
	#hits = 0

	# run the model to classify the documents
	#for item in dataset:
	#	output = model(item)
	#	predictedArgMax = output.data.cpu().numpy().argmax()
		#predicted = output.data.cpu().numpy()
	#	if predicted == 1:
	#		hits += 1
	#	if counter % 500 == 0:
	#		print(counter)
	#	counter = counter + 1
	#print("Total items classified: " + str(counter))
	#print("Total hits: " + str(hits))
	#pass
	#####################################################################################

def testAccuracy(curr_epoch, model, text_test, known_class_test):
	# test the model and output the accuracy
	accuracyList = []
	print("\nStarting testing for epoch " + str(curr_epoch))
	correct = 0
	counter = 0
	positive_correct = 0
	positive_count = 0
	for item in text_test:
		output = model(item)
		#predicted = output.data.cpu().numpy()
		predicted = output.data.cpu().numpy().argmax()
		#print("predicted class: " + str(predicted))
		#print("actual class: " + str(known_class_test[counter].numpy()))
		if predicted == known_class_test[counter].argmax():
			correct += 1
		if known_class_test[counter].argmax() == 1:
			positive_count += 1
			if predicted == known_class_test[counter].argmax():
				positive_correct += 1
		counter += 1

	accuracyList = [correct/counter * 100, positive_correct/positive_count * 100]
	print("\n\nAccuracy at epoch " + str(curr_epoch))
	print("Out of " + str(counter) + " articles the model predicted " + str(correct) +  
		  " correctly. " + str(round(correct/counter * 100, 7)) + "%")
	print("Out of " + str(positive_count) + " positive articles the model predicted " + 
	   str(positive_correct) + " correctly. " + str(round(positive_correct/positive_count * 100, 7)) + "%")

	return accuracyList


# this function adds a timestamp to the front of a filename
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S-{fname}'):
        import datetime
        # This creates a timestamped filename so we don't overwrite our good work
        return datetime.datetime.now().strftime(fmt).format(fname=fname)


# This function takes a file path, a list object for the texts, a list object 
# for the known class, and a classification value for the items in the file 
# It will append the text to the text list and the known class value to the 
# value list. The result will be two parallel lists containing the training and 
# test data. These lists will need to be sub-divided later into training and test data.
def fileToList(file_path, text_list, class_list, class_value):
	counter = 0
	one_counter = 0
	with open(file_path, encoding="utf-8") as f:
		for line in f:
			text_list.append(line)
			class_list.append(class_value)
			counter += 1
	pass
	

# This function takes the name of a txt file containing document ID numbers
# to be used to build a training set for the classifier and the path and
# filename of a csv file containing document IDs and the path to each 
# corresponding document (one doucment per record). The function reads the
# complete document index file, locates the documents in the txt file, extracts
# the path from the index file, and calls a function to read the file and add
# the text of each file to a list. Finally, it calls a function to build a NN
# model from the list. 
def indexToPath(textIDs, docIndex):
	# open the input file containing index values
	i = open(filename, 'r')
	
	# open the csv file correlating index values with file paths and read all of
	# the index value-filename pairs into a dictionary
	reader = csv.reader(open(docIndex, 'r'))
	d = {}
	for row in reader:
		k, v = row
		d[k] = v

	# set up the file path to the folder holding the twitter files
	twitter_stem = "<path to data folder goes here>\\TKX\\"

	# create a list to store all of the text from the twitter documents indicated by
	# the index values in the filename passed in. Each document is a single string
	# in the list
	nctt = []

	# get a line
	for line in i:
		# get rid of any white space
		index = line.strip()
		# ignore the first 13 lines which are not index values
		if int(index) > 13:
			# check to make sure it is not a blank line
			if len(index) > 0:
				# find this index in the master file 
				file_path = d.get(index)
				file_path = twitter_stem + file_path
				# copy the text from the file at this path into a list
				pathToList(file_path, nctt)
	# close the files
	i.close()
	
	listToModel(nctt)

# function to read the tweets from text files into a list of strings
#  commented this out for this application - also generates an csv file assigning 
#  an index value to each document
def filesToDocs(path, csvFilename):
    documents = []
    counter = 0
    #o = open("documentIndex.csv", 'w')
    o = open(csvFilename, 'w')
    for currentFile in glob2.glob(str(path) + '\\**\\*.txt'):
        #print("current file is: " + currentFile)
        i = open(currentFile, 'r', encoding = "utf-8")
        documents.append(i.read())
        # testing output
        #if counter >= 120:
        #    if counter < 122:
        #        print (currentFile)
        #        print ("\n")
        #        print(str(documents[counter]).encode("utf-8"))
        #        print("\n\n")
        i.close()
        o.write(str(counter) + "," + str(currentFile) + "\n" )
        counter = counter + 1
    print("counter = " + str(counter))
    o.close()
    return documents


# function to read the cosine similarities from a CSV file of document IDs and 
# cosine similarities and return a list of the similarities
# This function assumes that the CSV is composed of tuples of the form docID, similarity 
def CSVToList(csvFile):

	# read all of the cosine similarities into a list
	reader = csv.reader(open(csvFile, 'r'))
	d = []
	for row in reader:
		k, v = row
		d.append(float(v))
	return d
	

def buildModel(terror_negative, terror_positive):
#def buildModel(csvFile):
	# create and populate a list to hold all similarities
	#simList = CSVToList(csvFile)
	
	# convert list to numpy array
	#simArray = np.array(simList, dtype = float)
#	print(simArray)

	# use Otsu's technique to calculate a threshold to divide terror positive and terror negative documents
	#otsu(simArray)
	threshold = 0.3608 # used MatLab to calculate this value in a separate process

	# training text is the list that will contain the text to be classified
	training_text = []

	# training_target is the list that will contain the correct class of each line of text
	training_target = []

	# fileToList takes the name of the file containing the training text, a list 
	# to store the text, a list to store the true class for each line in the file, 
	# and the correct value of the text in the file
	#fileToList(sys.argv[1], training_text, training_target, 0) # non-terror-related items
	fileToList(terror_negative, training_text, training_target, 0) # non-terror-related items

	non_terror_item_count = len(training_text)
	print("non-terror items: " + str(non_terror_item_count))
	
	#fileToList(sys.argv[2], training_text, training_target, 1) # terror-related items
	fileToList(terror_positive, training_text, training_target, 1) # terror-related items

	#terror_item_count = len(training_text) - non_terror_item_count
	#print("terror items: " + str(terror_item_count))
	print("total items: " + str(len(training_text)))
	#print(training_target[166596:166600])
	listToModel(training_text, training_target)


# this function implements Otsu's method to calculate a histogram-based threshold for binarizing the dataset
def otsu(thresh_array):
	# set number of bins for histogram and threshold calculation
	n_bins = 256

	counts, edges = np.histogram(thresh_array, bins = n_bins)
	bin_centers = edges[:-1] + np.diff(edges) / 2

	total_ssds = []
	for bin_no in range(1, n_bins):
		left_ssd = ssd(counts[:bin_no], bin_centers[:bin_no])
		right_ssd = ssd(counts[bin_no:], bin_centers[bin_no:])
		total_ssds.append(left_ssd + right_ssd)
	z = np.argmin(total_ssds)
	t = bin_centers[z]
	print('Otsu bin (z):', z)
	print('Otsu threshold (c[z]):', bin_centers[z])
	return bin_centers[z]

# this is a utility function to calculate sum of squared deviations for Otsu's technique
def ssd(counts, centers):
	""" Sum of squared deviations from mean """
	n = np.sum(counts)
	mu = np.sum(centers * counts) / n
	return np.sum(counts * ((centers - mu) ** 2))


def fetchAndRunModel():

	# To load and use a previously trained model follow these steps
	dataset_text = filesToDocs("<path to data folder goes here>\\TKX\\twitter_screen_names_by_country", "terrorHits.csv")

	# load the model
	#model = TheModelClass(*args, **kwargs)
	input_length = 500
	output_length = 2

	# use this for 500 features
	h1, h2, h3, h4 = 243, 81, 27, 9
	#nn = nnModel(torch.nn.Linear(input_length, h1), 
	#				  torch.nn.ReLU(),
	#				  torch.nn.Linear(h1, h2),
	#				  torch.nn.ReLU(),
	#				  torch.nn.Linear(h2, h3),
	#				  torch.nn.ReLU(),
	#				  torch.nn.Linear(h3, h4),
	#				  torch.nn.ReLU(),
	#				  torch.nn.Linear(h4, output_length))
	#nn.load_state_dict(torch.load("NLP_RNN500_1597074319.140822.pth"))
	#optimizer = TheOptimizerClass(*args, **kwargs)
	#optimizer = torch.optim()
	#optimizer.load_state_dict()

	# provide the path to the file containing the model that you wish to load
	checkpoint = torch.load("NLP_RNN500_1597074319.140822.pth")

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	model.eval()
	# - or -
	#model.train()

	# dataset = the twitter data
	dataset = vectorizer.transform(dataset_text)
	dataset = torch.FloatTensor(dataset.toarray())

	# run the model to classify the documents
	for item in dataset:
		output = model(item)
		predicted = output.data.cpu().numpy().argmax()
		print(predicted)


def configModel(input_length, output_length):
	# specify the size of each hidden layer

	# use this for 5,000 features
	#h1, h2, h3, h4, h5, h6, h7, h8, h9 = 4000, 3000, 2000, 1000, 500, 243, 81, 27, 9

	# use this for 2,500 features
	#h1, h2, h3, h4, h5, h6 = 1000, 500, 243, 81, 27, 9

	# use this for 1,000 features
	#h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12 = 700, 500, 350, 125, 90, 60, 40, 13, 9, 6, 4, 2
	
	# use this for 500 features
	#h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13 = 350, 250, 167, 115, 90, 60, 40, 27, 18, 12, 8, 6, 3

	# use this for 250 features
	#h1, h2, h3, h4, h5, h6, h7, h8 = 150, 90, 54, 32, 19, 11, 6, 4

	# use this for 100 features
	#h1, h2, h3, h4, h5, h6, h7, h8 = 66, 40, 27, 18, 12, 8, 6, 4

	# use this for 50 features
	h1, h2, h3, h4, h5, h6, h7 = 33, 22, 15, 10, 7, 5, 3

	# use this for 30 features
	#h1, h2, h3, h4, h5 = 20, 12, 8, 5, 3

	# the output vectors should be the same length as the number of classes, in this case, 2 (passed in as parameter output_length)
	
	model = nn.Sequential(torch.nn.Linear(input_length, h1), 
					  torch.nn.ReLU(), 
					  torch.nn.Linear(h1, h2),
					  torch.nn.ReLU(),
					  torch.nn.Linear(h2, h3),
					  torch.nn.ReLU(),
					  torch.nn.Linear(h3, h4),
					  torch.nn.ReLU(),
					  torch.nn.Linear(h4, h5),
					  torch.nn.ReLU(),
					  torch.nn.Linear(h5, h6),
					  torch.nn.ReLU(),
					  torch.nn.Linear(h6, h7),
					  torch.nn.ReLU(),
					  #torch.nn.Linear(h7, h8),
					  #torch.nn.ReLU(),
					  #torch.nn.Linear(h8, h9),
					  #torch.nn.ReLU(),
					  #torch.nn.Linear(h9, h10),
					  #torch.nn.ReLU(),
					  #torch.nn.Linear(h10, h11),
					  #torch.nn.ReLU(),
					  #torch.nn.Linear(h11, h12),
					  #torch.nn.ReLU(),
					  #torch.nn.Linear(h12, h13),
					  #torch.nn.ReLU(),
					  torch.nn.Linear(h7, output_length),
					  )

	return model
	

if __name__ == '__main__':

	# to train a model run buildModel
	buildModel("<path to data folder goes here>\\Arabic_tweet_NN\\negative_correlation_terrorist_tweets_2021-03-02.txt", "<path to data folder goes here>\\Arabic_tweet_NN\\positive_correlation_terrorist_tweets_2021-03-02.txt")

	# to load and run a previously trained model run fetchAndRunModel
	#fetchAndRunModel()

