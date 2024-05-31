import os
import sys
import csv
import requests
from bs4 import BeautifulSoup
import json
import datetime
from datetime import date, timedelta
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
import time
import investpy
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from tensorflow.keras.models import load_model



#Load the csv file
df = pd.read_csv('data/processed_datav2.csv')
df.set_index('Date', inplace=True)
#Get the last updated date from df
last = df.index[-1]
last = tuple(map(int,last.split('-')))

#Functions for update the data
#BTC data
def get_btcdata():
    day = date.today() - timedelta(days=1)
    tday = date.today() + timedelta(days=1)
    data_lastday = datetime(*last).date()
    last_day = datetime(*last).date() + timedelta(days=1)

    if data_lastday < day:
        from_day = last_day.strftime("%d/%m/%Y")
        to_day = tday.strftime("%d/%m/%Y")

        btc_data = investpy.get_crypto_historical_data(crypto='bitcoin',
                                                from_date=from_day,
                                                to_date=to_day)

        btcdf = pd.DataFrame(btc_data)
        btcdf = btcdf.drop(columns=['Volume','Currency'])
        btcdf = btcdf.rename(columns={'Open':'btc_open','High':'btc_high','Low':'btc_low','Close':'btc_price'})
        df1 = pd.DataFrame(btcdf, columns=['btc_price', 'btc_open', 'btc_high', 'btc_low'])
        df1.fillna(method='ffill')
        df1.fillna(method='bfill')
        df1['btc_change']=round(((df1['btc_price']-df1['btc_open'])/df1['btc_open'])*100, 2)

        return df1
    
#Gold data
def get_golddata():
    day = date.today() - timedelta(days=1)
    tday = date.today() + timedelta(days=1)
    data_lastday = datetime(*last).date()
    last_day = datetime(*last).date() + timedelta(days=1)

    if data_lastday < day:
        from_day = last_day.strftime("%d/%m/%Y")
        to_day = tday.strftime("%d/%m/%Y")

        gold_data = investpy.get_commodity_historical_data(commodity='gold',
                                                   from_date=from_day,
                                                   to_date=to_day)

        golddf = pd.DataFrame(gold_data)
        golddf = golddf.drop(columns=['Volume','Currency'])
        golddf = golddf.rename(columns={'Open':'gold_open','High':'gold_high','Low':'gold_low','Close':'gold_price'})
        df2 = pd.DataFrame(golddf, columns=['gold_price', 'gold_open', 'gold_high', 'gold_low'])
        df2.fillna(method='ffill')
        df2.fillna(method='bfill')
        df2['gold_change']=round(((df2['gold_price']-df2['gold_open'])/df2['gold_open'])*100, 2)

        return df2

#S&P 500 data
def get_spdata():
    day = date.today() - timedelta(days=1)
    tday = date.today() + timedelta(days=1)
    data_lastday = datetime(*last).date()
    last_day = datetime(*last).date() + timedelta(days=1)

    if data_lastday < day:
        from_day = last_day.strftime("%d/%m/%Y")
        to_day = tday.strftime("%d/%m/%Y")

        sp500_data = investpy.get_index_historical_data(index='S&P 500',
                                                country='united states',
                                                from_date=from_day,
                                                to_date=to_day)

        spdf = pd.DataFrame(sp500_data)
        spdf = spdf.drop(columns=['Volume','Currency'])
        spdf = spdf.rename(columns={'Open':'sp_open','High':'sp_high','Low':'sp_low','Close':'sp_price'})
        df3 = pd.DataFrame(spdf, columns=['sp_price', 'sp_open', 'sp_high', 'sp_low'])
        df3.fillna(method='ffill')
        df3.fillna(method='bfill')
        df3['sp_change']=round(((df3['sp_price']-df3['sp_open'])/df3['sp_open'])*100, 2)

        return df3

#US30 data
def get_us30data():
    day = date.today() - timedelta(days=1)
    tday = date.today() + timedelta(days=1)
    data_lastday = datetime(*last).date()
    last_day = datetime(*last).date() + timedelta(days=1)

    if data_lastday < day:
        from_day = last_day.strftime("%d/%m/%Y")
        to_day = tday.strftime("%d/%m/%Y")

        us30_data = investpy.get_index_historical_data(index='Dow 30',
                                               country='united states',
                                               from_date=from_day,
                                               to_date=to_day)

        us30df = pd.DataFrame(us30_data)
        us30df = us30df.drop(columns=['Volume','Currency'])
        us30df = us30df.rename(columns={'Open':'us30_open','High':'us30_high','Low':'us30_low','Close':'us30_price'})
        df4 = pd.DataFrame(us30df, columns=['us30_price', 'us30_open', 'us30_high', 'us30_low'])
        df4.fillna(method='ffill')
        df4.fillna(method='bfill')
        df4['us30_change']=round(((df4['us30_price']-df4['us30_open'])/df4['us30_open'])*100, 2)

        return df4

#Dollar index data
def get_usdidxdata():
    day = date.today() - timedelta(days=1)
    tday = date.today() + timedelta(days=1)
    data_lastday = datetime(*last).date()
    last_day = datetime(*last).date() + timedelta(days=1)

    if data_lastday < day:
        from_day = last_day.strftime("%d/%m/%Y")
        to_day = tday.strftime("%d/%m/%Y")

        usdidx_data = investpy.get_index_historical_data(index='US Dollar Index',
                                               country='united states',
                                               from_date=from_day,
                                               to_date=to_day)

        usdidxdf = pd.DataFrame(usdidx_data)
        usdidxdf = usdidxdf.drop(columns=['Volume','Currency'])
        usdidxdf = usdidxdf.rename(columns={'Open':'usidx_open','High':'usidx_high','Low':'usidx_low','Close':'usidx_price'})
        df5 = pd.DataFrame(usdidxdf, columns=['usidx_price', 'usidx_open', 'usidx_high', 'usidx_low'])
        df5.fillna(method='ffill')
        df5.fillna(method='bfill')
        df5['usidx_change']=round(((df5['usidx_price']-df5['usidx_open'])/df5['usidx_open'])*100, 2)

        return df5



def fetch_data(get_data_func, *args, max_attempts=5, timeout=30):
    """
    Fetch data using the given function with retries and timeout.
    
    Args:
        get_data_func: Function to fetch data.
        *args: Arguments for the get_data_func.
        max_attempts (int): Maximum number of attempts.
        timeout (int): Timeout in seconds between attempts.
        
    Returns:
        DataFrame: Fetched data or None if unsuccessful.
    """
    for _ in range(max_attempts):
        try:
            data = get_data_func(*args)
            if not data.empty:
                return data
        except Exception as e:
            pass
        time.sleep(timeout)
    return None

def RSI(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

columns_of_interest = ['btc_price', 'btc_open', 'btc_high', 'btc_low', 'btc_change', 'gold_price', 'gold_open', 'gold_high', 'gold_low', 'gold_change', 
                       'sp_price', 'sp_open', 'sp_high', 'sp_low', 'sp_change', 'us30_price', 'us30_open', 'us30_high', 'us30_low', 'us30_change', 
                       'usidx_price', 'usidx_open', 'usidx_high', 'usidx_low', 'usidx_change', 'btc_rsi']



predictions_db = pd.read_csv('data/to_predict.csv')    
latest_date = predictions_db['Date'].iloc[-1]
latest_date_tuple = tuple(map(int,latest_date.split('-')))
latest_update = datetime(*latest_date_tuple).date()
loaded_model = load_model('models/dnn.h5')


if date.today() > latest_update:
    df1 = fetch_data(get_btcdata)
    df2 = fetch_data(get_golddata)
    df3 = fetch_data(get_spdata)
    df4 = fetch_data(get_us30data)
    df5 = fetch_data(get_usdidxdata)

    concatenated_df = pd.concat([df1,df2,df3,df4,df5],axis=1, join='outer')
    df_filled_forward = concatenated_df.fillna(method='ffill')
    df_filled = df_filled_forward.fillna(method='bfill')
    df_filled.index = df_filled.index.strftime('%Y-%m-%d')
    df_copy = df.copy()
    combined_df = pd.concat([df_copy, df_filled])
    combined_df['btc_rsi'] = RSI(combined_df['btc_price'])
    combined_df['btc_next'] = combined_df['btc_open'].shift(-1)
    to_predict = combined_df.iloc[-1][columns_of_interest].values
    new_df = combined_df.iloc[-10:, combined_df.columns.isin(columns_of_interest)]
    combined_df = combined_df.drop(combined_df.index[-1])
    combined_df['direction'] = combined_df.apply(lambda row: 1 if row['btc_next'] - row['btc_open'] > 0 else 0, axis=1)
    combined_df = combined_df.fillna(method='bfill')
    combined_df = combined_df.drop(columns='btc_next')
    
    combined_df.to_csv('data/processed_datav2.csv', index=True)
    new_df.to_csv('data/to_predict.csv', index=True)

    last_9_rows_values = combined_df.iloc[-9:][columns_of_interest].values
    last_10_rows_values = np.vstack([last_9_rows_values,to_predict])
    last_9_test = combined_df.iloc[-9:]['direction'].values

    

    predictions = loaded_model.predict(last_10_rows_values)
    y_pred_binary = np.round(predictions).astype(int)
    accuracy_last_9 = accuracy_score(last_9_test, y_pred_binary[:-1])

else:
    latest_data = pd.read_csv('data/to_predict.csv')
    latest_data.set_index('Date', inplace=True)
    price_data = pd.read_csv('data/processed_datav2.csv')
    price_data.set_index('Date', inplace=True)

    last_10_rows_values = latest_data.iloc[-10:][columns_of_interest].values
    last_9_test = price_data.iloc[-9:]['direction'].values
    predictions = loaded_model.predict(last_10_rows_values)
    y_pred_binary = np.round(predictions).astype(int)
    accuracy_last_9 = accuracy_score(last_9_test, y_pred_binary[:-1])



    

    





    


