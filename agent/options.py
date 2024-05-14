import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


SYMBOL = "WTI"
SHARES_100 = 100
NUM_WRITE = 2
ERROR = 0.07

def get_options_data(ticker, historical_date):
    ''' Uses Yahoo Finance to get all upcoming options data including contractSymbol, strike, impliedVolatility, type, expiration date'''

    options_chain = ticker.option_chain(historical_date)
    options_df = pd.concat([options_chain.calls, options_chain.puts])

    # create type column and expiration date column
    options_df["premium"] = (options_df["bid"] + options_df["ask"]) / 2
    options_df["optionType"] = options_df['contractSymbol'].str[9].apply(lambda x: 'Call' if x == 'C' else 'Put')
    options_df["expirationDate"] = options_df['contractSymbol'].str[3:9].apply(lambda x: (datetime.strptime(x, "%y%m%d")).strftime("%m-%d-%Y"))


    # drop everything we prolly dont need
    options_df = options_df.drop(["lastTradeDate", "lastPrice", "volume", "bid", "ask", "change", "percentChange", "openInterest"], axis=1)
    options_df = options_df[options_df['contractSize'] == 'REGULAR']    # get rid of others
    options_df = options_df[options_df['currency'] == 'USD']        # get rid of others
    options_df = options_df.drop(["contractSize", "currency"], axis=1)
    
    # put these columns as the last ones cause of my OCD
    itm_column = options_df.pop("inTheMoney")
    options_df["inTheMoney"] = itm_column
    volatility_column = options_df.pop("impliedVolatility")
    options_df["impliedVolatility"] = volatility_column

    #reset index
    options_df.reset_index(drop=True, inplace=True)

    return options_df

def get_straddles(call_options_df, put_options_df):
    '''Get all possible straddles that can be created from options data'''
    
    # initialize straddles dataframe
    straddles = pd.DataFrame(columns=["strike", "expirationDate", "callPremium", "putPremium", "totalCost", "lowerBound", "upperBound"])

    # get puts available to make straddle with and update straddles accordingly
    for index, call_row in call_options_df.iterrows():
        strike = call_row["strike"]
        expiration = call_row["expirationDate"]
        corresponding_put = put_options_df[(put_options_df["expirationDate"] == expiration) & (put_options_df["strike"] == strike)]
        if not corresponding_put.empty:
            corresponding_put = corresponding_put.iloc[0]
            straddle_row = {
                "strike": strike,
                "expirationDate": expiration,
                "callPremium": call_row["premium"],
                "putPremium": corresponding_put["premium"],
                "totalCost": call_row["premium"] + corresponding_put["premium"],
                "lowerBound": strike - (call_row["premium"] + corresponding_put["premium"]),
                "upperBound": strike + (call_row["premium"] + corresponding_put["premium"]),
            }
            straddles = pd.concat([straddles, pd.DataFrame([straddle_row])], ignore_index=True)

    return straddles


def calc_straddle_profit(my_straddle, curr_price):
    '''Calculate profit for a given straddle trade and current price'''
    if curr_price > my_straddle["strike"]:      # in call range
        return SHARES_100*curr_price - SHARES_100*my_straddle["strike"]
    elif curr_price < my_straddle["strike"]:  # in put range
        return -SHARES_100*curr_price + SHARES_100*my_straddle["strike"]
    else:
        return 0
    

def get_butterflies(call_options_df):
    '''Get all possible butterflies that can be created from options data'''
    itm_calls = call_options_df[call_options_df["inTheMoney"] == True]
    otm_calls = call_options_df[call_options_df["inTheMoney"] == False]

    # get the calls to write by getting the strike closest to the current price that is ITM
    max_itm_strike = itm_calls['strike'].max()

    calls_to_write = itm_calls[itm_calls['strike'] == max_itm_strike]

    #reset index
    calls_to_write.reset_index(drop=True, inplace=True)

    # initialize straddles dataframe
    possible_butterflies = pd.DataFrame(columns=["writeStrike", "writePremium", "expirationDate", "itmCallPremium", "imtCallStrike", 
                                                 "otmCallPremium", "otmCallStrike", "totalCost", "lowestBound", "lowerBound", "upperBound", "uppestBound",])
    
    for index, write_row in calls_to_write.iterrows():
        for index2, itm_call_row in itm_calls.iterrows():
            otm_call = otm_calls[(otm_calls["expirationDate"] == write_row["expirationDate"]) & (otm_calls["expirationDate"] == itm_call_row["expirationDate"]) &
                                            (otm_calls["strike"] - write_row["strike"] == write_row["strike"] - itm_call_row["strike"])] # butterfly must be perfect!!
            if not otm_call.empty:
                otm_call = otm_call.iloc[0]
                butterfly_row = {
                    "writeStrike": write_row["strike"],
                    "writePremium": write_row["premium"],
                    "expirationDate": write_row["expirationDate"],
                    "itmCallPremium": itm_call_row["premium"],
                    "imtCallStrike": itm_call_row["strike"],
                    "otmCallPremium": otm_call["premium"],
                    "otmCallStrike": otm_call["strike"],
                    "totalCost": itm_call_row["premium"] + otm_call["premium"] - NUM_WRITE*write_row["premium"],
                    "lowestBound": itm_call_row["strike"], 
                    "lowerBound": itm_call_row["strike"] + (itm_call_row["premium"] + otm_call["premium"] - 2*write_row["premium"]), 
                    "upperBound": otm_call["strike"] -  + (itm_call_row["premium"] + otm_call["premium"] - 2*write_row["premium"]), 
                    "uppestBound": otm_call["strike"],
                }
                possible_butterflies = pd.concat([possible_butterflies, pd.DataFrame([butterfly_row])], ignore_index=True)

    return possible_butterflies


def calc_butterfly_profit(my_butterfly, curr_price):
    '''Calculate profit for a given butterfly trade and current price'''
    itm_call_profit = (SHARES_100*curr_price) - SHARES_100*my_butterfly['imtCallStrike']
    otm_call_profit = (SHARES_100*curr_price) - SHARES_100*my_butterfly['otmCallStrike']
    write_call_profit = -((SHARES_100*curr_price) - SHARES_100*my_butterfly['writeStrike'])*NUM_WRITE
    # total_cost = SHARES_100 * my_butterfly["totalCost"]

    if curr_price >= my_butterfly["upperBound"] or curr_price <= my_butterfly["lowerBound"]:
        return 0
    elif curr_price > my_butterfly["writeStrike"]:
        return itm_call_profit + write_call_profit
    elif curr_price < my_butterfly["writeStrike"]:
        return itm_call_profit


def find_best_option(curr_price, actual_var, predicted_var, call_options_df, put_options_df):
    ''' Selects butterfly or straddle based on predicted. Then gets the option within the possible ones
        that minimized the distance to the current_price'''
    option_type = None
    if predicted_var > actual_var + ERROR:      # our model expects higher variance --> straddle
        option_type = "Straddle"
        special_options = get_straddles(call_options_df, put_options_df)   
    elif predicted_var < actual_var - ERROR:     # our model expects lower variance --> butterfly
        option_type = "Butterfly"
        special_options = get_butterflies(call_options_df)
    else:
        return None, None
    
    # Find the row with lowerBound and upperBound columns closest to the current price
    best_option = None
    lowest_diff = None
    for index, row in special_options.iterrows():
        curr_diff = abs(row['lowerBound'] - curr_price) + abs(row['upperBound'] - curr_price)
        if best_option is None:
            best_option = row
            lowest_diff = curr_diff
        elif curr_diff < lowest_diff:
            best_option = row
            lowest_diff = curr_diff
    
    return best_option, option_type
    

def get_best_option(predicted_var, actual_var, historical_date, curr_price):

    ticker = yf.Ticker(SYMBOL)


    options_df = get_options_data(ticker, historical_date)

    # sort by expiration date
    options_df['expirationDate'] = pd.to_datetime(options_df['expirationDate'])
    options_df = options_df[options_df['expirationDate'].dt.dayofweek > 2]
    options_df = options_df.sort_values(by='expirationDate')

    # we will only trade off the closest day after Wednesday so get that date and only use trades with that day
    closest_date = options_df.iloc[0]["expirationDate"]
    options_df = options_df[options_df["expirationDate"]==closest_date]

    # get calls and puts 
    call_options_df = options_df[options_df['optionType'] == 'Call']
    put_options_df = options_df[options_df['optionType'] == 'Put']

    # reset their indices
    call_options_df = call_options_df.reset_index(drop=True)
    put_options_df = put_options_df.reset_index(drop=True)

    # print(curr_price)
    # print(actual_var)
    # print(predicted_var)
    # print(call_options_df)
    # print(put_options_df)
    option_trade, option_type = find_best_option(curr_price, actual_var, predicted_var, call_options_df, put_options_df)

    if option_trade is None:
        return None, None
    else:
        return option_trade, option_type