import pandas as pd
SPLIT_DATE = "2020-04-29"
SPLIT_RATIO = 8

'''Create csv of uso prices that are not adjusted for the stock split'''
    
if __name__ == '__main__':
    uso_data = pd.read_csv('./USO.csv', index_col="Date")
    uso_data['Non-Adj Close'] = uso_data['Adj Close']
    uso_data.loc[uso_data.index < SPLIT_DATE, 'Non-Adj Close'] /= SPLIT_RATIO
    uso_data = uso_data[['Non-Adj Close','Adj Close']]
    uso_data.to_csv('./USO_updated.csv', index=True)