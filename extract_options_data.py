import pandas as pd
import warnings
import requests

warnings.filterwarnings("ignore")

STRIKE_BOUND = 0.10
SPLIT_DATE = "2020-04-29"
SPLIT_RATIO = 8

def create_non_adjusted_options(cs_data, uso_data, url):
    ''' Get historical options data from MarketData for the dates that we are using the cloud score from.
    Only grabbing options symbol, expiration date, option type, premium, strike, and itm. This funntion 
    does NOT adjust the price of the options, just like MarketData. https://www.marketdata.app/docs/api/options/chain'''
    all_data = pd.DataFrame()

    for date, cs_row in cs_data.iterrows():
    
        if date in uso_data.index:
            uso_price = uso_data.loc[date]["Non-Adj Close"]
            bound = str(uso_price-(uso_price*STRIKE_BOUND)) + "-" + str(uso_price+(uso_price*STRIKE_BOUND))

            parameter = {"date":date, "strike":bound, "weekly":True, "token":key}

            date_response = requests.get(url, params=parameter)

            if date_response.status_code == 200:
                data = date_response.json()
                for option_symbol, dte, premium, strike, in_the_money, side in zip(data['optionSymbol'], data['dte'], data['mid'], data['strike'], data['inTheMoney'], data['side']):
                    expiration_date = ((pd.Timestamp(date) + pd.Timedelta(days=dte)).date()).strftime("%Y-%m-%d")
                    wanted_data = {
                        'dateFind': date,
                        'optionSymbol': option_symbol,
                        'expirationDate': expiration_date,
                        'optionType': side,
                        'premium': premium,
                        'strike': strike,
                        'inTheMoney': in_the_money
                    }
                    all_data = all_data._append(wanted_data, ignore_index=True)
            else:
                print("Failed. Status code:", date_response.status_code)

    return all_data



def create_adjusted_options(cs_data, uso_data, url):
    ''' Get historical options data from MarketData for the dates that we are using the cloud score from.
    Only grabbing options symbol, expiration date, option type, premium, strike, and itm. This funntion adjusts
    the price of the options to handle the stock split since MarketData does not. https://www.marketdata.app/docs/api/options/chain'''
    all_data = pd.DataFrame()

    for date, cs_row in cs_data.iterrows():
    
        if date in uso_data.index:
            uso_price = uso_data.loc[date]["Non-Adj Close"]
            bound = str(uso_price-(uso_price*STRIKE_BOUND)) + "-" + str(uso_price+(uso_price*STRIKE_BOUND))

            parameter = {"date":date, "strike":bound, "weekly":True, "token":key}

            date_response = requests.get(url, params=parameter)

            if date_response.status_code == 200:
                data = date_response.json()
                for option_symbol, dte, premium, strike, in_the_money, side in zip(data['optionSymbol'], data['dte'], data['mid'], data['strike'], data['inTheMoney'], data['side']):
                    if date < SPLIT_DATE:
                        expiration_date = ((pd.Timestamp(date) + pd.Timedelta(days=dte)).date()).strftime("%Y-%m-%d")
                        wanted_data = {
                            'dateFind': date,
                            'optionSymbol': option_symbol,
                            'expirationDate': expiration_date,
                            'optionType': side,
                            'premium': premium * SPLIT_RATIO,
                            'strike': strike * SPLIT_RATIO,
                            'inTheMoney': in_the_money
                        }
                        all_data = all_data._append(wanted_data, ignore_index=True)
                    else:
                        expiration_date = ((pd.Timestamp(date) + pd.Timedelta(days=dte)).date()).strftime("%Y-%m-%d")
                        wanted_data = {
                            'dateFind': date,
                            'optionSymbol': option_symbol,
                            'expirationDate': expiration_date,
                            'optionType': side,
                            'premium': premium,
                            'strike': strike,
                            'inTheMoney': in_the_money
                        }
                        all_data = all_data._append(wanted_data, ignore_index=True)
            else:
                print("Failed. Status code:", date_response.status_code)

    return all_data
    

if __name__ == '__main__':
    uso_data = pd.read_csv('./USO_updated.csv', index_col="Date")

    key = 'aHZ1OTJEY2FTdVRYOVUzazdZcUQ2Y0FzcHU4YjR3eVJxNnVOZDRkc2RtND0'
    url = "https://api.marketdata.app/v1/options/chain/USO"

    cs_data = pd.read_csv('./combined.csv', index_col="Unnamed: 0")

    # CHOSE WHAT U WANT
    # non_adj_data = create_non_adjusted_options(cs_data, uso_data, url)
    # non_adj_data.to_csv('./historical_options_uso_non_adj.csv', index=False)

    adj_data = create_adjusted_options(cs_data, uso_data, url)
    adj_data.to_csv('./historical_options_uso_adj.csv', index=False)
