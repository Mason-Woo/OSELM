"""
Online Sequential ELM (OS-ELM) ver 1.0
Based on the paper: Liang N., Huang G., Saratchedran P., Sundararajan N.,(2006 November), 
A Fast and Accurate Online Sequential Learning Algorithm for Feedforward Networks, 
IEE Transactions on Neural Networks, Vol. 17, No.6
@author:batli
	
"""
#complete packages
import pdb
import os, sys
import numpy as np

#import from modules
eps = np.finfo(float).eps
from six import integer_types
from scipy.spatial.distance import cdist
from sklearn.preprocessing import label_binarize, LabelBinarizer

def softmax(x):
	#Compute softmax values for each sets of scores in x.
	assert len(x.shape) == 2
	xs = np.max(x, axis=1)
	xs = xs[:, np.newaxis] # necessary step to do broadcasting
	e_x = np.exp(x - xs)
	div = np.sum(e_x, axis=1)
	div = div[:, np.newaxis] # dito
	return e_x / div

class OSELM():
	def __init__(self, inputs, outputs, classification = "r", alpha = 1.0, binarizer=LabelBinarizer(-1, 1)):
		assert isinstance(inputs, integer_types), "Number of inputs must be integer"
		assert isinstance(outputs, integer_types), "Number of outputs must be integer"
		self.inputs = inputs	     #input size of the input vector
		self.outputs = outputs    #output size of the output vector
		self.neurons = []
		self.N = 0	#number of neurons
		self.B = None	#bias
		self.H = None	#hidden layer output matrix
		self.HH = None	#H*H.T
		self.HT = None	#H.T

		#initialize the classification properties
		self.dELM = None
		self.classes = outputs
		self.binarizer = LabelBinarizer() #binarizer that will be used during classification in multiple class cases 
		self.alpha = alpha #constant that regularizes the radius distance of one class classification
		self.classification = classification 
		#classification('c')/regression('r')/one class classification('oc')

		# transformation functions
		self.flist = ("lin", "sigm", "tanh", "rbf_l1", "rbf_l2", "hardlim")  # supported neuron types

	def actFunc(self, func, X, W, B):
		if func == "lin":
			G = np.dot(X, W) + B
		elif func == "sigm":
			G = 1 / (1 + np.exp(np.dot(X, W) + B))
		elif func == "tanh":
			G = np.tanh(np.dot(X, W) + B)
		elif func == "rbf_l1":
			G = np.exp(B*(-cdist(X, W.T, "cityblock")**2))
		elif func == "rbf_l2":
			G = np.exp(B*(-cdist(X, W.T, "euclidean")**2))
		elif func == "hardlim":
			G = np.maximum(np.sign(np.dot(X,W) + B + self.epsilon), np.zeros((len(X),len(W))))
		return G
	
	def _checkdata(self, X, T):
		#Checks data variables and fixes matrix dimensionality issues.
		assert len(self.neurons) > 0, "Add neurons to ELM before training it"
		if X is not None:
			if len(X.shape) == 1:
				X = X.reshape(-1, 1)
			assert X.shape[1] == self.inputs, "X has wrong dimensionality: expected %d, found %d" %(self.inputs, X.shape[1])
		if T is not None:
			if(len(T.shape) == 1):
				T = T.reshape(-1, 1)
			assert X.shape[1] == self.inputs, "T has wrong dimensionality: expected %d, found %d" %(self.outputs, T.shape[1])
		return X, T

	def add_neurons(self, number, func, W=None, B=None):
		#Add neurons and initialize weights/biases
		#Weights/biases can also be given in the function as a biased learning
		assert (func in self.flist), "Neuron type is not supported, activation functions: " + str(self.flist)
		inputs = self.inputs
		if W is None:
			if func == "lin":
				number = min(number, inputs)  #linear neurons can not be more than size of input features
				W = np.eye(inputs, number)
			else:
				W = np.random.normal(size=(inputs, number))
		else:
			if len(W.shape) == 1:
				W = np.vstack([W]*number).T
			elif len(W.shape) == 2 and W.shape[1] == 1:
				W = W.reshape(inputs,)
				W = np.vstack([W]*number).T
			elif len(W.shape) == 2 and W.shape[1] == 2:
				W = W
		if B is None:
			weight_scale = 0.5
			B = weight_scale*np.random.normal(size=(number,)) + 0.1 # the original was np.random.normal(size=(number,))
		if func in ("rbf_l1", "rbf_l2"): #make beta positive 
                	B = abs(B)

		ftypes = [n[1] for n in self.neurons]  # existing types of neurons
		if func in ftypes:
			# add to an existing neuron type
			i = ftypes.index(func)
			nn0, func0, W0, B0 = self.neurons[i]
			W = np.hstack((W0, W))
			B = np.hstack((B0, B))
			number = nn0 + number
			self.neurons[i] = (number, func, W, B)
		else:
			# create a new neuron type
			self.neurons.append((number, func, W, B))
		self.N += number
		#clear learning parameters if new type of neurons added
		self.B  = None
		self.P  = None 
		self.HH = None
		self.HT = None

	def binarize(self, T):
		# 2 classes in multiclassifier input
		if self.outputs == 2:
			T = self.binarizer.fit_transform(T)
			return np.hstack((1-T, T))
		#more than one classes in multiclassifier input
		else:
			return (self.binarizer.fit_transform(T))
	def classify(self, YH):
		#for multiple classes and classification case, function binarizes the predicted data Y 
		for i in range(len(YH)):
			row = YH[i]
			idx  = np.argmax(row)
			row[:]=0
			row[idx]=1
		return self.binarizer.inverse_transform(YH) 

	def regressor_fit(self, X0, T0):
		#initialize the learning using first chunk of initial training data from training set.
		X0, T0 = self._checkdata(X0,T0)
		self.HH = np.zeros((self.N , self.N))
		self.HT = np.zeros((self.N, self.outputs))
		H = np.hstack([self.actFunc(ftype, X0, W, B) for _, ftype, W, B in self.neurons]) 
		self.HH += np.dot(H.T, H)
		self.HT += np.dot(H.T, T0)
		HH_pinv = np.linalg.pinv(self.HH)
		B = np.dot(HH_pinv, self.HT)
		self.B = B
		self.H = H 

	def classifier_fit(self, X0, T0):
		self.regressor_fit(X0, T0)
		HB = self.regressor_predict(X0)
		#original computation
		#self.dELM = abs(HB - 1).max() #TODO is HB.mean() can be changed with class label y(=1)?
		self.dELM = ((HB-1)**2).max()*self.alpha
	
	def fit(self, X0, T0):
		#fit the supervised data 
		if self.classification == 'r':
			self.regressor_fit(X0, T0)
		elif self.classification == 'c':
			T0 = self.binarize(T0)
			self.regressor_fit(X0, T0)
		elif self.classification == 'oc':
			self.classifier_fit(X0, T0)

	def regressor_add_data(self, XN, TN):
		XN, TN = self._checkdata(XN, TN)
		inputs = len(TN)
		#weights for the next chunk
		HN = np.hstack([self.actFunc(ftype, XN, W, B) for _, ftype, W, B in self.neurons]) 

		P =  np.linalg.pinv(self.HH + np.dot(HN.T, HN))
		bias  = self.B + reduce(np.dot, [P, HN.T, (TN - HN.dot(self.B))])

		self.HH = np.dot(HN.T, HN)
		self.HT = np.dot(HN.T, TN)
		self.B = bias

	def classifier_add_data(self, XN, TN):
		self.regressor_add_data(XN, TN)
		HB = self.regressor_predict(XN)
		self.dELM = ((HB-1)**2).max()*self.alpha
		#original distance computation abs(HB - 1).max()

	def add_data(self, XN, TN):
		#sequential learning phase when (k+1)th chunk of new observations presented to the learning  
		if self.classification == 'r':
			self.regressor_add_data(XN, TN)
		elif self.classification == 'c':
			TN = self.binarize(TN)
			self.regressor_add_data(XN, TN)
		elif self.classification == 'oc':
			self.classifier_add_data(XN, TN)

	def regressor_predict(self, X):
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		H = np.hstack([self.actFunc(ftype, X, W, B) for _, ftype, W, B in self.neurons]) 
		Y = np.dot(H, self.B)
		return Y

	def classifier_predict(self, X):
		Y = self.regressor_predict(X)
		#original computation
		#return np.sign(self.dELM - abs(1-Y))
		return np.sign(self.dELM - (1-Y)**2 + eps)

	def predict(self, X):
		if self.classification == 'r':
			Y = self.regressor_predict(X)
		elif self.classification == 'c':
			Y = self.regressor_predict(X)
			Y = self.classify(Y)
		elif self.classification == 'oc':
			Y = self.classifier_predict(X)
		return Y

	def decision_function(self, X):
		#decision function values for each sample as distance of samples to the separating hyperplane
		Yr = self.regressor_predict(X)
		#original distance func
		#Yc = self.dELM - abs(1-Yr)
		Yc = (1-Yr)**2 - self.dELM #np.maximum(0, (1-Yr)**2 - self.dELM) #self.dELM - (1-Yr)**2
		return Yc

	def predict_proba(self, X):
		#Probability estimates. The returned estimates for all classes are ordered by the label of classes.
		Ypred = self.regressor_predict(X)	
		#if self.classes == 2:
		#	return np.vstack([1 - Ypred, Ypred]).T
		#else:
		return softmax(Ypred)
		
 
            
