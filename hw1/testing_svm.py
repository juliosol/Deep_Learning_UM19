import numpy as np
from solver import *
from layers import *
import pickle
from logistic import *
from svm import *
#import pandas as pd

with open('D:/Julio/Documents/Michigan_v2/CS/EECS_598_Deep_Learning/HW/hw1/Homework1/code/data.pkl', 'rb') as f:
  data = pickle.load(f, encoding='latin1')

features = data[0]
labels = data[1]

training_features = features[0:500, :]
training_labels = labels[0:500]


validation_features = features[500:750, :]
validation_labels = labels[500:750]

test_features = features[750:1000,:]
test_labels = labels[750:1000]

### Loading data
data = {
    'X_train': training_features, # training data
    'y_train': training_labels, # training labels
    'X_val': validation_features, # validation data
    'y_val': validation_labels # validation labels
}

data_test = {
    'X_test': test_features,
    'y_test': test_labels
}

input_dim = training_features.shape[1]

## Loading model and training

model = SVM(hidden_dim=1, input_dim=input_dim, reg=0.005)

solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 0.999,
                  },
                  lr_decay=0.95,
                  num_epochs=1000, batch_size=15,
                  print_every=100)
solver.train()

print("Test accuracy SVM " + str(solver.check_accuracy(test_features, test_labels)))
