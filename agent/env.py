import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import options as op

warnings.filterwarnings("ignore")

SHARES_100 = 100
SD_TIME = 7
STARTING_CASH = 10000
NUM_TRIPS = 500
HOLDINGS = 1000
NUM_TRIALS = 10

class LinearNN(nn.Module):
    def __init__(self, num_features, output_size):
        super(LinearNN, self).__init__()
        self.relu = nn.ReLU()
        self.input_to_layer_1 = nn.Linear(num_features, 64)       #.double() 
        self.layer_1_to_layer_2 = nn.Linear(64, 10)     #.double()        
        self.layer_2_to_output = nn.Linear(10, output_size)     #.double() 
        self.fake = nn.Linear(num_features, output_size)

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

    def train_env(self, data, uso_data):

        for date, row_cs in data.iterrows():
            # fast forward one week and one day in time and get the close! (Wednesday close) --> want this because the price should have readjusted
            # date index is the start of the week, which is why we go a week and a day in future
            next_day = ((pd.Timestamp(date) + pd.Timedelta(days=8)).date()).strftime("%Y-%m-%d")
            next_week = ((pd.Timestamp(date) + pd.Timedelta(days=SD_TIME)).date()).strftime("%Y-%m-%d")
            if date in uso_data.index and next_day in uso_data.index and next_week in uso_data.index:

                row_cs_tensor = torch.tensor(row_cs.values, dtype=torch.float)

                y_actual = uso_data.loc[next_week]["Volatility"]  #CHECK ME: is next week right?
                self.learner.train(row_cs_tensor, y_actual)
    
        return


    def test_env(self, data, uso_data, options_data_uso, in_sample):
        cash_over_time = []
        cash_dates = []
        cash = STARTING_CASH
        for date, row_cs in data.iterrows():
            # fast forward one week and one day in time and get the close! (Wednesday close) --> want this because the price should have readjusted
            # date index is the start of the week, which is why we go a week and a day in future
            next_day = ((pd.Timestamp(date) + pd.Timedelta(days=8)).date()).strftime("%Y-%m-%d")
            next_week = ((pd.Timestamp(date) + pd.Timedelta(days=SD_TIME)).date()).strftime("%Y-%m-%d")
            if date in uso_data.index and next_day in uso_data.index and next_week in uso_data.index:
                
                row_cs_tensor = torch.tensor(row_cs.values, dtype=torch.float)

                y_actual = uso_data.loc[next_week]["Volatility"]  #CHECK ME: is next week right?
                y_pred = self.learner.test(row_cs_tensor)

                # get the best option
                curr_price = uso_data.loc[next_week]["Adj Close"]
                option_to_trade, option_type = op.get_best_option(y_pred, y_actual, options_data_uso, next_week, curr_price)

                if option_to_trade is not None: # case where there are no possible straddles/butterflies
                    # get price of next day
                    next_day_close = (uso_data.loc[next_day])["Adj Close"]
                    
                    # if we have enough cash to make trade, do it!
                    total_cost = option_to_trade['totalCost']*SHARES_100
                    if cash - total_cost >= 0:
                        if option_type == "Butterfly":
                            cash += op.calc_butterfly_profit(option_to_trade, next_day_close)
                        elif option_type == "Straddle":
                            cash += op.calc_straddle_profit(option_to_trade, next_day_close)
                        cash -= total_cost
                cash_over_time.append(cash/STARTING_CASH)
                cash_dates.append(date)


        # BASE LINE
        uso_data_filtered = uso_data.loc[cash_dates]

        initial_price = uso_data_filtered.iloc[0]['Adj Close']
        initial_investment = HOLDINGS * initial_price

        investment_values = HOLDINGS * uso_data_filtered['Adj Close']
        baseline_return = ((investment_values - initial_investment) / initial_investment) + 1

        cash_dates = pd.to_datetime(cash_dates)

        # plt.plot(cash_dates, cash_over_time, label='My Portfolio', color="royalblue")
        # plt.plot(cash_dates, baseline_return, label='Baseline Portfolio', color="darkorange")     
        # plt.xlabel("Date")
        # plt.ylabel("Cumulative Return")
        # if in_sample:
        #     plt.title("Cumulative Return over Time -- In Sample")
        # else:
        #     plt.title("Cumulative Return over Time -- Out of Sample")
        # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))  # Set the interval to display ticks every month
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format the tick labels as year-month
        # plt.xticks(rotation=45)
        # plt.legend()
        # plt.show()

        return cash/STARTING_CASH, baseline_return
    


if __name__ == '__main__':

    # read in csvs
    data = pd.read_csv('./combined.csv', index_col="Unnamed: 0")

    uso_data = pd.read_csv('./USO_updated.csv', index_col="Date")
    uso_data["Volatility"] = uso_data["Adj Close"].rolling(SD_TIME).std()

    options_data_uso = pd.read_csv('./historical_options_uso_2.csv')

    # get shapes
    num_rows = data.shape[0]
    num_features = data.shape[1]

    # get the first 2/3 of the rows
    cutoff_row = int(num_rows * 2 / 3)
    train_data = data.iloc[:cutoff_row]
    test_data = data.iloc[cutoff_row:]

    is_cr = []
    oos_cr = []
    is_bench = []
    oos_bench = []
    for i in range(NUM_TRIALS):
        # create learner and network
        env = CloudLearner() 

        env.learner = LinearNNLearner(learning_rate=0.001, epochs=5, num_features=num_features, output_size=1)
        
        for i in range(NUM_TRIPS):
            print("Trip number: " + str(i))
            env.train_env(train_data, uso_data)
            env.learner.losses_trips.append(np.mean(env.learner.losses))
            env.learner.losses = []

        is_final_cash, baseline_is = env.test_env(train_data, uso_data, options_data_uso, in_sample=True)
        is_cr.append(is_final_cash)
        is_bench.append(baseline_is)

        oos_final_cash, baseline_oos = env.test_env(test_data, uso_data, options_data_uso, in_sample=False)
        oos_cr.append(oos_final_cash)
        oos_bench.append(baseline_oos)


    is_cr = np.array(is_cr)
    oos_cr = np.array(oos_cr)

    # Print summary results.
    print ()
    print (f"In-sample per-symbol per-day min, median, mean, max results across all {NUM_TRIALS} trials")
    print(f"IS : {np.min(is_cr):.4f}, {np.median(is_cr):.4f}, {np.mean(is_cr):.4f}, {np.max(is_cr):.4f} vs long benchmark {np.mean(is_bench):.4f}")

    print ()
    print (f"Out-of-sample per-symbol per-day min, median, mean, max results across all {NUM_TRIALS} trials")
    print(f"OOS: {np.min(oos_cr):.4f}, {np.median(oos_cr):.4f}, {np.mean(oos_cr):.4f}, {np.max(oos_cr):.4f} vs long benchmark {np.mean(oos_bench):.4f}")


