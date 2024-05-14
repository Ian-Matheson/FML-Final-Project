import options as op
import yfinance as yf
import pandas as pd

SYMBOL = "WTI"
ERROR = 0.05
STARTING_CASH = 10000
SHARES_100 = 100

def select_option(curr_price, actual_var, predicted_var, call_options_df, put_options_df):

    if predicted_var > actual_var + ERROR:      # our model expects higher variance --> straddle
        special_options = op.get_straddles(call_options_df, put_options_df)   
    elif predicted_var < actual_var - ERROR:     # our model expects lower variance --> butterfly
        special_options = op.get_butterflies(call_options_df)
    else:
        return None
    
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
    
    return best_option
    

if __name__ == "__main__":

    ticker = yf.Ticker(SYMBOL)
    curr_price = ticker.history(period='1d')['Close'].iloc[-1]

    options_df = op.get_options_data(ticker)

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


    actual_var = 0.2
    predicted_var = 0.1

    option_trade = select_option(curr_price, actual_var, predicted_var, call_options_df, put_options_df)

    if option_trade is None:
        print("FAIL")
