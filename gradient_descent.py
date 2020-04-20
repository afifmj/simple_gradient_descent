from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
	return 1.0/(1+np.exp(-x))

def predict(X,W):
	#Taking dot product between features ans weight matrices
	preds = sigmoid_activation(X.dot(W))

	#We threshold the predictions to binary
	preds[preds <= 0.5] = 0
	preds[preds > 0.5] = 1

	return preds

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
help="learning rate")
args = vars(ap.parse_args())

#generating a dataset with 1000 data points and 2 features
(X,y) = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std = 1.5, random_state = 1)
#X -> [no. of sample, no. of features]
#y ->[no. of samples] //row vector containing the label for each sample

#reshaping y to a column vector 
y = y.reshape((y.shape[0],1))

#apply the bias trick, ie, add a column of ones to the X matrix
X = np.c_[X, np.ones((X.shape[0]))]

#splitting the training and testing data to 50-50%
(trainX, testX , trainY, testY) = train_test_split(X,y,test_size = 0.5, random_state=42)

print("[INFO] training...")

#randomly initializing the weight matrix using uniform distribution and haveing the same dimensions as X
W = np.random.randn(X.shape[1], 1)

#a list to keep track of loss after each epoch(which was taken as input)
losses = []

#looping through each epoch, we do the actual gradient descent
for epoch in np.arange(0,args['epochs']):

	#get the label prediction for each feature in our training data
	preds = sigmoid_activation(trainX.dot(W))

	#find the error in prediction by comparing with actual label in the training data
	error = preds-trainY
	loss = np.sum(error**2)
	losses.append(loss)

	#find the gradient 
	gradient = trainX.T.dot(error)

	#doing the actual gradient descent
	W += -args['alpha'] * gradient

	# check to see if an update should be displayed
	if epoch == 0 or (epoch + 1) % 5 == 0:
		print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),loss))

#Done with training, its time to evaluate/test our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
