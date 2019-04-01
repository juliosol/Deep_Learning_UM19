import numpy as np
from solver import *
from layers import *
import pickle
from logistic import *
#import pandas as pd

with open('D:/Julio/Documents/Michigan_v2/CS/EECS_598_Deep_Learning/HW/hw1/Homework1/code/data.pkl', 'rb') as f:
  data = pickle.load(f, encoding='latin1')

features = data[0]
labels = data[1]

training_features = features[0:500, :]
training_labels = labels[0:500]
#training_labels = training_labels.reshape(-1,1)


validation_features = features[500:750, :]
validation_labels = labels[500:750]
#validation_labels = validation_labels.reshape(-1,1)

test_features = features[750:1000,:]
test_labels = labels[750:1000]
#test_labels = test_labels.reshape(-1,1)

### Loading data
data = {
    'X_train': training_features, # training data
    'y_train': training_labels, # training labels
    'X_val': validation_features, # validation data
    'y_val': validation_labels # validation labels
}

#input_dim = training_features.shape[1]
input_dim=20

## Loading model and training

model = LogisticClassifier(hidden_dim=1, input_dim=input_dim, reg=0.05)

solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 0.9999
                  },
                  lr_decay=0.99,
                  num_epochs=2000, batch_size=20,
                  print_every=100)
solver.train()

print("test accuracy " + str(solver.check_accuracy(test_features, test_labels)))
