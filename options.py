import yfinance as yf
import pandas as pd
from datetime import datetime


SYMBOL = "WTI"
SHARES_100 = 100

def get_options_data(ticker):
    ''' Uses Yahoo Finance to get all upcoming options data including contractSymbol, strike, impliedVolatility, type, expiration date'''
    options_dates = ticker.options

    # initialize
    options_df = pd.DataFrame()

    # create DataFrame for non-expired options
    for date in options_dates:
        options_chain = ticker.option_chain(date)
        options_df = pd.concat([options_df, options_chain.calls, options_chain.puts])

    # create type column and expiration date column
    options_df["premium"] = (options_df["bid"] + options_df["ask"]) / 2
    options_df["optionType"] = options_df['contractSymbol'].str[9].apply(lambda x: 'Call' if x == 'C' else 'Put')
    options_df["expirationDate"] = options_df['contractSymbol'].str[3:9].apply(lambda x: (datetime.strptime(x, "%y%m%d")).strftime("%m-%d-%Y"))


    # drop everything we prolly dont need
    options_df = options_df.drop(["lastTradeDate", "lastPrice", "volume", "bid", "ask", "change", "percentChange", "openInterest", "inTheMoney"], axis=1)
    options_df = options_df[options_df['contractSize'] == 'REGULAR']    # get rid of others
    options_df = options_df[options_df['currency'] == 'USD']        # get rid of others
    options_df = options_df.drop(["contractSize", "currency"], axis=1)
    
    #reset index
    options_df.reset_index(drop=True, inplace=True)

    return options_df

def get_straddles(options_df):
    '''Get all possible straddles that can be created from options data'''
    options_df = options_df.copy()

    # get calls and puts 
    call_options_df = options_df[options_df['optionType'] == 'Call']
    put_options_df = options_df[options_df['optionType'] == 'Put']

    # reset their indices
    call_options_df = call_options_df.reset_index(drop=True)
    put_options_df = put_options_df.reset_index(drop=True)

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
        return SHARES_100*curr_price - SHARES_100*my_straddle["strike"] - SHARES_100*my_straddle["totalCost"]
    elif curr_price < my_straddle["strike"]:  # in put range
        return -SHARES_100*curr_price + SHARES_100*my_straddle["strike"] - SHARES_100*my_straddle["totalCost"]
    else:
        return -SHARES_100*my_straddle["totalCost"]


if __name__ == "__main__":
    ticker = yf.Ticker(SYMBOL)
    current_price = ticker.history(period='1d')['Close'].iloc[-1]

    options_df = get_options_data(ticker)
    straddles = get_straddles(options_df)

    #say we choose straddle 0 and we cash out now:
    straddle_profit = calc_straddle_profit(straddles.iloc[0], current_price)
    # print(current_price)
    # print(options_df)
    # print(straddles)
