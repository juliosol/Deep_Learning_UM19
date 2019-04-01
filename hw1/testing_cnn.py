import numpy as np
from solver import *
from layers import *
from softmax import *
import pickle
import torch
import torchvision
import matplotlib.pyplot as plt
from cnn import *

import gzip, numpy

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(np.shape(x_train))
#print(np.shape(x_test))
#print(np.shape(y_train))
#print(np.shape(y_test))

training_pictures = x_train[0:50000, :, :]
#training_pictures = training_pictures.reshape(50000, -1)

training_labels = y_train[0:50000]

validation_pictures = x_train[50000:60000, :, :]
#validation_pictures = validation_pictures.reshape(10000, -1)

validation_labels = y_train[50000:60000]

testing_pictures = x_test
testing_pictures = testing_pictures.reshape(10000, -1)
testing_labels = y_test


### Loading data
data = {
    'X_train': training_pictures, # training data
    'y_train': training_labels, # training labels
    'X_val': validation_pictures, # validation data
    'y_val': validation_labels # validation labels
}

input_dim = training_pictures.shape
print(input_dim)
#input_dim=20

## Loading model and training

model = ConvNet(reg=0.05)

solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 9e-8
                  },
                  lr_decay=0.95,
                  num_epochs=5, batch_size=15,
                  print_every=100)
solver.train()

#print(np.shape(x _test))
#plt.imshow(x_train[5,:,:], cmap='Greys')
#plt.show()
