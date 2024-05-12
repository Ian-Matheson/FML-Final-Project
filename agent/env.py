import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import warnings
# warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

class RNN(nn.Module):
    def __init__(self, num_features, hidden_layer_size, output_size):
        super(RNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.rnn = nn.RNN(num_features, hidden_layer_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, x):
        hidden_state = torch.zeros(1, x.size(0), self.hidden_layer_size)    # initialize hidden state with zeros
    
        out, _ = self.rnn(x, hidden_state)      # forward propagate RNN

        out = self.output_layer(out[:, -1, :])      # pass the output of the last time step through the fully connected layer
        return out

class CloudLearner:

    def __init__(self, learning_rate=0.001, epochs=5, num_features=8000,
                 hidden_layer_size=128, output_size=1):    # Plus whatever parameters you need.
        self.device = torch.device("cpu")
        
        self.network = RNN(num_features, hidden_layer_size, output_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

        self.num_epochs = epochs

    def train(self, data):

        for date, cloud_score in data.items():
            
            self.network.train()
            self.optimizer.zero_grad()
            y = np.random.rand(1) # NEED TO GET ACTUAL VOLATILITY
            batch_loss = self.loss_func(self.network(cloud_score), y)
            batch_loss.backward()
            self.optimizer.step()
            batch_loss = batch_loss.item()
            

    def test(self, data):

        for date, cloud_score in data.items():
            with torch.no_grad():
                variance = self.network(cloud_score)

        return variance

def cloud_score(start_date, end_date, df, num_features):
    # make me real!
    data = {}

    data[start_date] = np.random.rand(num_features) #give random (fake) cloud scores per feature
    data[end_date] = np.random.rand(num_features) #give random (fake) cloud scores per feature

    return data


if __name__ == '__main__':

    # select dates
    start_date = "2020-01-01"
    end_date = "2022-01-01"

    # read in csv
    df = pd.read_csv('FRT_coords/tank_inventory.csv')
    num_features = df.shape[0] # number of rows is number of features. RIGHT?

    # should pass in start and end date into cloud score retriever
    # data will be dictionary, {date: (f1, f2, ..., f8000)}
    # DO we want this to also have the actual volitility at that time?
    train_data = cloud_score(start_date, end_date, df, num_features)
    print(train_data)

    # create learner and network
    env = CloudLearner(learning_rate=0.001, epochs=5, num_features=num_features,
                 hidden_layer_size=128, output_size=1) 
    

    num_trips = 100
    for i in range(num_trips):
        env.train(train_data)

    # need different data
    test_data = cloud_score(start_date, end_date, df, num_features)
    env.test(test_data)

