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
import sys
import options as op

SHARES_100 = 100
SD_TIME = 7
STARTING_CASH = 10000
NUM_TRIPS = 50

class LinearNN(nn.Module):
    def __init__(self, num_features, output_size):
        super(LinearNN, self).__init__()
        self.relu = nn.ReLU()
        self.input_to_layer_1 = nn.Linear(num_features, 256)       #.double() 
        self.layer_1_to_layer_2 = nn.Linear(256, 128)     #.double()        
        self.layer_2_to_output = nn.Linear(128, output_size)     #.double() 


    def forward(self, x):
        x = self.input_to_layer_1(x)
        x = self.relu(x)
        x = self.layer_1_to_layer_2(x)
        x = self.relu(x)
        x = self.layer_2_to_output(x)
        return x
    

class LinearNNLearner(nn.Module):
    def __init__(self, num_features=8000, learning_rate=0.001, epochs=5, output_size=1):    # Plus whatever parameters you need.
        super(LinearNNLearner, self).__init__()  # Call the __init__() method of the parent class
        self.device = torch.device("cpu")
        
        self.network = LinearNN(num_features, output_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        self.num_epochs = epochs
        self.losses = []
        self.losses_trips = []
        
        
    def train(self, row_cs, y_actual):
        self.network.train()
        self.optimizer.zero_grad()

        y_pred = self.network(row_cs)
        batch_loss = self.loss_func(y_pred, torch.tensor(y_actual, dtype=torch.float))

        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()
        self.losses.append(batch_loss)

        return y_pred.item()
    
    def test(self, row_cs):
        with torch.no_grad():
            y_pred = self.network(row_cs)

        return y_pred.item()

class CloudLearner:

    def __init__(self):    # Plus whatever parameters you need.
        self.learner = None

    def train_env(self, data, wti_data):

        cash = STARTING_CASH
        for date, row_cs in data.iterrows():
            # fast forward one week and one day in time and get the close! (Wednesday close) --> want this because the price should have readjusted
            # date index is the start of the week, which is why we go a week and a day in future
            next_day = ((pd.Timestamp(date) + pd.Timedelta(days=8)).date()).strftime("%Y-%m-%d")
            next_week = ((pd.Timestamp(date) + pd.Timedelta(days=SD_TIME)).date()).strftime("%Y-%m-%d")
            if date in wti_data.index and next_day in wti_data.index and next_week in wti_data.index:
                row_cs_tensor = torch.tensor(row_cs.values, dtype=torch.float)

                y_actual = wti_data.loc[next_week]["Volatility"]  #CHECK ME: is next week right?
                y_pred = self.learner.train(row_cs_tensor, y_actual)

                # get price of next day
                option_to_trade, option_type = op.get_best_option(y_pred, y_actual)
                
                if option_to_trade is not None:  # case where there are no possible straddles/butterflies
                    # fast forward one week and one day in time and get the close! (Wednesday close) --> want this because the price should have readjusted
                    next_day_close = (wti_data.loc[next_day])["Adj Close"]

                    # if we have enough cash to make trade, do it!
                    total_cost = option_to_trade['totalCost']*SHARES_100
                    if cash - total_cost >= 0:
                        if option_type == "Butterfly":
                            cash += op.calc_butterfly_profit(option_to_trade, next_day_close)
                        if option_type == "Straddle":
                            cash += op.calc_straddle_profit(option_to_trade, next_day_close)
                        cash -= total_cost*SHARES_100
        
        # plt.plot(range(len(env.learner.losses)), env.learner.losses)
        # plt.xlabel("Iterations")
        # plt.ylabel("Loss")
        # plt.title("Training Loss over Iterations")
        # plt.show()
        return


    def test_env(self, data, wti_data):

        cash_over_time = []
        cash_dates = []
        cash = STARTING_CASH
        for date, row_cs in data.iterrows():
            # fast forward one week and one day in time and get the close! (Wednesday close) --> want this because the price should have readjusted
            # date index is the start of the week, which is why we go a week and a day in future
            next_day = ((pd.Timestamp(date) + pd.Timedelta(days=8)).date()).strftime("%Y-%m-%d")
            next_week = ((pd.Timestamp(date) + pd.Timedelta(days=SD_TIME)).date()).strftime("%Y-%m-%d")
            if date in wti_data.index and next_day in wti_data.index and next_week in wti_data.index:
                row_cs_tensor = torch.tensor(row_cs.values, dtype=torch.float)

                y_actual = wti_data.loc[next_week]["Volatility"]  #CHECK ME: is next week right?
                with torch.no_grad():
                    y_pred = self.learner.test(row_cs_tensor)

                # get the best option
                option_to_trade, option_type = op.get_best_option(y_pred, y_actual)

                if option_to_trade is not None: # case where there are no possible straddles/butterflies
                    # get price of next day
                    next_day_close = (wti_data.loc[next_day])["Adj Close"]
                    
                    # if we have enough cash to make trade, do it!
                    total_cost = option_to_trade['totalCost']*SHARES_100
                    if cash - total_cost >= 0:
                        if option_type == "Butterfly":
                            cash += op.calc_butterfly_profit(option_to_trade, next_day_close)
                        if option_type == "Straddle":
                            cash += op.calc_straddle_profit(option_to_trade, next_day_close)
                        cash -= total_cost

                cash_over_time.append(cash/STARTING_CASH)
                cash_dates.append(date)

        plt.plot(cash_dates, cash_over_time)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.title("Cumulative Return over Time")
        # plt.xticks(rotation=45)
        # plt.grid(True)
        plt.show()

        return cash
    


if __name__ == '__main__':

    # read in csvs
    data = pd.read_csv('./combined.csv', index_col="Unnamed: 0")
    wti_data = pd.read_csv('./WTI.csv', index_col="Date")
    wti_data = wti_data[['Adj Close']]
    wti_data["Volatility"] = wti_data["Adj Close"].rolling(SD_TIME).std()

    # get shapes
    num_rows = data.shape[0]
    num_features = data.shape[1]

    # Get the first 2/3 of the rows

    cutoff_row = int(num_rows * 2 / 3)
    train_data = data.iloc[:cutoff_row]
    test_data = data.iloc[cutoff_row:]

    # create learner and network
    env = CloudLearner() 

    env.learner = LinearNNLearner(learning_rate=0.001, epochs=5, num_features=num_features, output_size=1)
    
    
    for i in range(NUM_TRIPS):
        print("Trip number: " + str(i))
        env.train_env(train_data, wti_data)
        env.learner.losses_trips.append(np.mean(env.learner.losses))
        env.learner.losses = []

    plt.plot(range(len(env.learner.losses_trips)), env.learner.losses_trips)
    plt.xlabel("Trips")
    plt.ylabel("Loss")
    plt.title("Training Loss over 50 Trips -- Significant Deviation=0.07, Variance Time Frame (days)=7, Starting Cash=10000" )
    plt.show()
    
    final_cash = env.test_env(train_data, wti_data)
    print("FINAL CASH: " + str(final_cash))
    print("CUMULATIVE RETURN: " + str(final_cash/STARTING_CASH))

