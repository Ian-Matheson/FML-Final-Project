import yfinance as yf
import pandas as pd
from datetime import datetime


SYMBOL = "CL"

def get_options_data():
    ''' Uses Yahoo Finance to get all upcoming options data including contractSymbol, strike, impliedVolatility, type, expiration date'''
    ticker = yf.Ticker(SYMBOL)
    options_dates = ticker.options

    # initialize
    options_df = pd.DataFrame()

    # create DataFrame for non-expired options
    for date in options_dates:
        options_chain = ticker.option_chain(date)
        options_df = pd.concat([options_df, options_chain.calls, options_chain.puts])


    # drop everything we prolly dont need
    options_df = options_df.drop(["lastTradeDate", "lastPrice", "bid", "ask", "volume", "change", "percentChange", "openInterest", "inTheMoney"], axis=1)
    options_df = options_df[options_df['contractSize'] == 'REGULAR']    # get rid of others
    options_df = options_df[options_df['currency'] == 'USD']        # get rid of others
    options_df = options_df.drop(["contractSize", "currency"], axis=1)

    # create type column and expiration date column
    options_df["optionType"] = options_df['contractSymbol'].str[8].apply(lambda x: 'Call' if x == 'C' else 'Put')
    options_df["expirationDate"] = options_df['contractSymbol'].str[2:8].apply(lambda x: (datetime.strptime(x, "%y%m%d")).strftime("%m-%d-%Y"))

    #reset index
    options_df.reset_index(drop=True, inplace=True)

    return options_df

if __name__ == "__main__":
    options_df = get_options_data()
    print(options_df)
