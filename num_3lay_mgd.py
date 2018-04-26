# Numerai EdNet
#
# 3 layer net with minibatch gradient descent
#
# Eduardo Bermudez

#!/usr/bin/env python
import csv
import pandas as pd
import math
import random
import numpy as np
import struct,string

class EdNet:
    def __init__(self, W, W2, W3, b, b2, b3):
        self.W=W
        self.W2=W2
        self.W3=W3
        self.b=b
        self.b2=b2
        self.b3=b3

def get_data(filename,includes_category): # input file converted into data matrix - size will be num examples x num features
    features = pd.read_csv(filename,dtype=float,usecols=list(range(3,53)))
    categories = np.zeros((features.shape[0],1))
    if includes_category:
      categories = pd.read_csv(filename,dtype=float,usecols=list(range(53,54)))
    categories=categories.astype(int)
    return [features.values, categories.values]

def run_net(data,net): # this will take in the data matrix and will produce probabily spaces for each example
    W=net.W
    W2=net.W2
    W3=net.W3
    b=net.b
    b2=net.b2
    b3=net.b3

    layer_1 = np.maximum(0, np.dot(data,W) + b)
    layer_2 = np.maximum(0, np.dot(layer_1,W2) + b2)
    net_scores = np.dot(layer_2,W3) + b3

    exp_net_scores = np.exp(net_scores)
    prob_spaces = exp_net_scores / np.sum(exp_net_scores, axis=1, keepdims=True)

    return prob_spaces

def validate_net(train_data_attributes, train_data_correct_classes, validation_data, validation_data_correct_classes): # this will take in a net and validation data and will produce ?
    net = net_learning(train_data, train_data_correct_classes,50,2)
    prob_spaces = run_net (validation_data, net)

    predictions = np.zeros(1,validation_data_correct_classes.shape[0])
    for i in xrange(validation_data_correct_classes.shape[0]):
      if prob_spaces[i][1]>prob_spaces[i][0]:
        predictions[0][i] = 1

    num_correct = np.dot(predictions, validation_data_correct_classes)
    accuracy = num_correct/validation_data_correct_classes.shape[0]
    return accuracy

def evaluate(filename, net):
  features = pd.read_csv(filename,dtype=float,usecols=list(range(3,53)))
  ids = pd.read_csv(filename,dtype=str,usecols=list(range(0,1)))

  W = net.W
  W2 = net.W2
  W3 = net.W3
  b = net.b
  b2 = net.b2
  b3 = net.b3

  hidden_layer_1 = np.maximum(0, np.dot(features,W) + b)
  hidden_layer_2 = np.maximum(0, np.dot(hidden_layer_1,W2) + b2)
  scores = np.dot(hidden_layer_2,W3) + b3

  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  results =probs[:,1]

  df=pd.DataFrame(results,columns = ['probability'])

  return pd.concat([ids,df],axis=1)

def print_results(results):
  results.to_csv('output.csv',encoding='utf-8',index=False)
  return True

def net_learning(X,y,D,K): # this will take in training data (with classifiers) and will return the net

#D = 	50 # dimensionality
#K = 	2  # number of classes
#y=[correct classes] 	# for training, equal to 1 when correct class, equal to zero otherwise. One-dimensional array size is number of classes  x num_examples 
#X=[] 					# array of attributes. size is [num_examples x number of attributes] = [num_examples x D]

# initialize parameters randomly
    h = 100 
    W = .2 * np.random.randn(D,h)		# D is dimensionality... .2 = sqrt(2/n) - He et al
    b = np.zeros((1,h))
    W2 = .2 * np.random.randn(h,h)	# h is the size of the hidden layers
    b2 = np.zeros((1,h))
    W3 = .2 * np.random.randn(h,K)	# K is the number of classes
    b3 = np.zeros((1,K))
    
    # some hyperparameters
    step_size = 1e-1
    reg = 1e-4# regularization strength
    print "test"
    
    # gradient descent loop
    num_examples = X.shape[0]
    num_train = 500000

    loss_func = np.zeros((2,num_train))
    for i in xrange(num_train):

      # Minibatch gradient descent - using 512 examples
      index = np.random.randint(num_examples-1,size=1024)
      batch = X[index,:]
      batch_y = y[index,:]
      
      # evaluate class scores, [N x K]
      hidden_layer_1 = np.maximum(0, np.dot(batch,W) + b)
      hidden_layer_2 = np.maximum(0, np.dot(hidden_layer_1,W2) + b2)
      scores = np.dot(hidden_layer_2,W3) + b3
      
      # compute the class probabilities
      exp_scores = np.exp(scores)
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
      
      # compute the loss: average cross-entropy loss and regularization
      correct_logprobs = -np.log(probs[range(1024),batch_y.T])
      data_loss = np.sum(correct_logprobs)/1024
      reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2) + 0.5*reg*np.sum(W3*W3)
      loss = data_loss + reg_loss
      if i % 2000 == 0:
        print "iteration %d: loss %f" % (i, np.mean(loss_func[1][i-99:i]))
      
      loss_func[0][i]=i
      loss_func[1][i]=loss
      #if i %10000==0:
      #  print probs[:50]
      #  print batch_y[:50]

      # compute the gradient on scores
      dscores = probs
      dscores[range(1024),batch_y.T] -= 1
      dscores /= 1024
      
      # backpropate the gradient to the parameters
      # first backprop into parameters W3 and b3
      dW3 = np.dot(hidden_layer_2.T, dscores)
      db3 = np.sum(dscores, axis=0, keepdims=True)
    
      dhidden_2 = np.dot(dscores, W3.T)
      dhidden_2[hidden_layer_2 <= 0 ] = 0 # backprop ReLU non-linearity
    
      #then backprop onto b2 and W2
      dW2 = np.dot(hidden_layer_1.T, dhidden_2)
      db2 = np.sum(dhidden_2, axis=0, keepdims=True)
    
      dhidden_1 = np.dot(dhidden_2, W2.T)
      dhidden_1[hidden_layer_1 <= 0] = 0 #again, backprop the ReLU non-linearity
    
      # Lastly, backprop onto b and W 
      dW = np.dot(batch.T, dhidden_1)
      db = np.sum(dhidden_1, axis=0, keepdims=True)
      
      # add regularization gradient contribution
      dW3 += reg * W3
      dW2 += reg * W2
      dW += reg * W
      
      # perform a parameter update
      W += -step_size * dW
      b += -step_size * db
      W2 += -step_size * dW2
      b2 += -step_size * db2
      W3 += -step_size * dW3
      b3 += -step_size * db3

    net = EdNet(W,W2,W3,b,b2,b3)
    return net

Data = get_data('data/numerai_training_data.csv',True)
Ednet=net_learning(Data[0],Data[1],50,2)
results = evaluate('data/numerai_tournament_data.csv',Ednet)
print_results(results)

