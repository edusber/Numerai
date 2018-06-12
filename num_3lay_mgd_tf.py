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
import argparse
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD



def get_data(filename,includes_category): # input file converted into data matrix - size will be num examples x num features
    features = pd.read_csv(filename,dtype=float,usecols=list(range(3,53)))
    categories = np.zeros((features.shape[0],1))
    if includes_category:
      categories = pd.read_csv(filename,dtype=float,usecols=list(range(53,54)))
    categories=categories.astype(int)
    return [features.values, categories.values]

Data = get_data('data/numerai_training_data.csv',True)
validation = get_data('data/validation.csv',True)
tournament = pd.read_csv('data/numerai_tournament_data.csv',dtype=float,usecols=list(range(3,53)))

# Generate dummy data
x_train = Data[0]
y_train = Data[1]

## Need to identify test data (live data)
x_test = validation[0]
y_test = validation[1]

model = Sequential()
model.add(Dense(100, input_dim=50, activation='relu'))
model.add(Dropout(0.1)) ### what is this? - reg loss?
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='relu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False) ## test different hyperparams
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size=1024)
score = model.evaluate(x_test, y_test, batch_size=128)

# then evaluate on unclassified data and print to file output.csv

predictions = model.predict(tournament)
output = pd.DataFrame(predictions,columns = ['probability']).to_csv('output.csv', index = False)




