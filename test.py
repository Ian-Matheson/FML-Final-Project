import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import requests

STRIKE_BOUND = 0.10
SPLIT_DATE = "2020-04-29"
SPLIT_RATIO = 8


if __name__ == '__main__':
    uso_data = pd.read_csv('./USO.csv', index_col="Date")
    uso_data['Non-Adj Close'] = uso_data['Adj Close']
    uso_data.loc[uso_data.index < SPLIT_DATE, 'Non-Adj Close'] /= SPLIT_RATIO
    uso_data = uso_data[['Non-Adj Close','Adj Close']]
    # uso_data.to_csv('./USO_updated.csv', index=True)

    key = 'aHZ1OTJEY2FTdVRYOVUzazdZcUQ2Y0FzcHU4YjR3eVJxNnVOZDRkc2RtND0'
    url = "https://api.marketdata.app/v1/options/chain/USO"

    cs_data = pd.read_csv('./combined.csv', index_col="Unnamed: 0")

    all_data = pd.DataFrame()

    # for date, cs_row in cs_data.iterrows():
    
    #     if date in uso_data.index:
    #         uso_price = uso_data.loc[date]["Non-Adj Close"]
    #         bound = str(uso_price-(uso_price*STRIKE_BOUND)) + "-" + str(uso_price+(uso_price*STRIKE_BOUND))

    #         parameter = {"date":date, "strike":bound, "weekly":True, "token":key}

    #         date_response = requests.get(url, params=parameter)

    #         if date_response.status_code == 200:
    #             data = date_response.json()
    #             for option_symbol, dte, premium, strike, in_the_money, side in zip(data['optionSymbol'], data['dte'], data['mid'], data['strike'], data['inTheMoney'], data['side']):
    #                 if date < SPLIT_DATE:
    #                     expiration_date = ((pd.Timestamp(date) + pd.Timedelta(days=dte)).date()).strftime("%Y-%m-%d")
    #                     wanted_data = {
    #                         'dateFind': date,
    #                         'optionSymbol': option_symbol,
    #                         'expirationDate': expiration_date,
    #                         'optionType': side,
    #                         'premium': premium * 8,
    #                         'strike': strike * 8,
    #                         'inTheMoney': in_the_money
    #                     }
    #                     all_data = all_data._append(wanted_data, ignore_index=True)
    #                 else:
    #                     expiration_date = ((pd.Timestamp(date) + pd.Timedelta(days=dte)).date()).strftime("%Y-%m-%d")
    #                     wanted_data = {
    #                         'dateFind': date,
    #                         'optionSymbol': option_symbol,
    #                         'expirationDate': expiration_date,
    #                         'optionType': side,
    #                         'premium': premium,
    #                         'strike': strike,
    #                         'inTheMoney': in_the_money
    #                     }
    #                     all_data = all_data._append(wanted_data, ignore_index=True)
    #         else:
    #             print("Failed. Status code:", date_response.status_code)

    # all_data.to_csv('./historical_options_uso_2.csv', index=False)
