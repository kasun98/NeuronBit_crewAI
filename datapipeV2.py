import pandas as pd
import numpy as np
import time
import requests
import datetime
from datetime import date, datetime, timedelta
from datetime import datetime
import investpy
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

class DataProcessor:
    def __init__(self, data_file='data/processed_datav2.csv', predict_file='data/to_predict.csv', btc_file='data/bitcoin.csv', model_path='models/dnn.h5', latest_accuracy='data/latest_accuracy.csv', pred_results = 'data/predicted_results.csv'):
        self.data_file = data_file
        self.predict_file = predict_file
        self.btc_file = btc_file
        self.model_path = model_path
        self.latest_accuracy = latest_accuracy
        self.pred_results = pred_results


        self.df = pd.read_csv(data_file)
        self.df.set_index('Date', inplace=True)
        self.latest_date = self.df.index[-1]
        self.latest_date_tuple = tuple(map(int, self.latest_date.split('-')))
        self.latest_update = datetime(*self.latest_date_tuple).date()


        self.predict_db = pd.read_csv(predict_file)
        self.predict_db.set_index('Date', inplace=True)
        self.latest_predicted_date = self.predict_db.index[-1]
        self.latest_predicted_date_tuple = tuple(map(int, self.latest_predicted_date.split('-')))
        self.latest_predicted_date_update = datetime(*self.latest_predicted_date_tuple).date()




        self.columns_of_interest = ['btc_price', 'btc_open', 'btc_high', 'btc_low', 'btc_change', 'gold_price', 'gold_open', 'gold_high', 'gold_low', 'gold_change', 
                                    'sp_price', 'sp_open', 'sp_high', 'sp_low', 'sp_change', 'us30_price', 'us30_open', 'us30_high', 'us30_low', 'us30_change', 
                                    'usidx_price', 'usidx_open', 'usidx_high', 'usidx_low', 'usidx_change', 'btc_rsi']

    def fetch_data(self, get_data_func, *args, max_attempts=25, timeout=5):
        for _ in range(max_attempts):
            try:
                data = get_data_func(*args)
                if not data.empty:
                    a = 'Data collected :) '
                    print(a)
                    return data
            except Exception as e:
                a = 'Fetch error! :('
            time.sleep(timeout)
        print(a)
        return None
    
    def is_weekend(self, d):
    # weekday() method returns 5 for Saturday and 6 for Sunday
        if d.weekday() >= 5:
            return 1
        else:
            return 0
    
    # Data getting functions
    def get_btcdata(self):
        day = date.today() - timedelta(days=1)
        tday = date.today() + timedelta(days=1)
        data_lastday = datetime(*self.latest_date_tuple).date()
        last_day = datetime(*self.latest_date_tuple).date() + timedelta(days=1)

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
            df1.ffill()
            df1.bfill()
            df1['btc_change']=round(((df1['btc_price']-df1['btc_open'])/df1['btc_open'])*100, 2)

            return df1

    def get_golddata(self):
        day = date.today() - timedelta(days=1)
        tday = date.today() + timedelta(days=1)
        data_lastday = datetime(*self.latest_date_tuple).date()
        last_day = datetime(*self.latest_date_tuple).date() + timedelta(days=1)

        if data_lastday < day:
            from_day = last_day.strftime("%d/%m/%Y")
            to_day = tday.strftime("%d/%m/%Y")

            d1 = last_day + timedelta(days=1)
            d2 = date.today()
            if d1 == d2 and self.is_weekend(d1)==1 and self.is_weekend(d2)==1:
                data_z = {
                    'Date': [last_day, d1],
                    'gold_price': self.df.iloc[-1]['gold_price'],
                    'gold_open': self.df.iloc[-1]['gold_open'],
                    'gold_high': self.df.iloc[-1]['gold_high'],
                    'gold_low': self.df.iloc[-1]['gold_low'],
                    'gold_change':self.df.iloc[-1]['gold_change']}

                
                df2 = pd.DataFrame(data_z)
                df2['Date'] = pd.to_datetime(df2['Date'])
                df2.set_index('Date', inplace=True)

                return df2
            else:


                gold_data = investpy.get_commodity_historical_data(commodity='gold',
                                                        from_date=from_day,
                                                        to_date=to_day)

                golddf = pd.DataFrame(gold_data)
                golddf = golddf.drop(columns=['Volume','Currency'])
                golddf = golddf.rename(columns={'Open':'gold_open','High':'gold_high','Low':'gold_low','Close':'gold_price'})
                df2 = pd.DataFrame(golddf, columns=['gold_price', 'gold_open', 'gold_high', 'gold_low'])
                df2.ffill()
                df2.bfill()
                df2['gold_change']=round(((df2['gold_price']-df2['gold_open'])/df2['gold_open'])*100, 2)

                return df2

    def get_spdata(self):
        day = date.today() - timedelta(days=1)
        tday = date.today() + timedelta(days=1)
        data_lastday = datetime(*self.latest_date_tuple).date()
        last_day = datetime(*self.latest_date_tuple).date() + timedelta(days=1)

        if data_lastday < day:
            from_day = last_day.strftime("%d/%m/%Y")
            to_day = tday.strftime("%d/%m/%Y")

            d1 = last_day + timedelta(days=1)
            d2 = date.today()
            if d1 == d2 and self.is_weekend(d1)==1 and self.is_weekend(d2)==1:
                data_z = {
                    'Date': [last_day, d1],
                    'sp_price': self.df.iloc[-1]['sp_price'],
                    'sp_open': self.df.iloc[-1]['sp_open'],
                    'sp_high': self.df.iloc[-1]['sp_high'],
                    'sp_low': self.df.iloc[-1]['sp_low'],
                    'sp_change':self.df.iloc[-1]['sp_change']}

                
                df3 = pd.DataFrame(data_z)
                df3['Date'] = pd.to_datetime(df3['Date'])
                df3.set_index('Date', inplace=True)

                return df3
            else:


                sp500_data = investpy.get_index_historical_data(index='S&P 500',
                                                        country='united states',
                                                        from_date=from_day,
                                                        to_date=to_day)

                spdf = pd.DataFrame(sp500_data)
                spdf = spdf.drop(columns=['Volume','Currency'])
                spdf = spdf.rename(columns={'Open':'sp_open','High':'sp_high','Low':'sp_low','Close':'sp_price'})
                df3 = pd.DataFrame(spdf, columns=['sp_price', 'sp_open', 'sp_high', 'sp_low'])
                df3.ffill()
                df3.bfill()
                df3['sp_change']=round(((df3['sp_price']-df3['sp_open'])/df3['sp_open'])*100, 2)

                return df3

    def get_us30data(self):
        day = date.today() - timedelta(days=1)
        tday = date.today() + timedelta(days=1)
        data_lastday = datetime(*self.latest_date_tuple).date()
        last_day = datetime(*self.latest_date_tuple).date() + timedelta(days=1)

        if data_lastday < day:
            from_day = last_day.strftime("%d/%m/%Y")
            to_day = tday.strftime("%d/%m/%Y")

            d1 = last_day + timedelta(days=1)
            d2 = date.today()
            if d1 == d2 and self.is_weekend(d1)==1 and self.is_weekend(d2)==1:
                data_z = {
                    'Date': [last_day, d1],
                    'us30_price': self.df.iloc[-1]['us30_price'],
                    'us30_open': self.df.iloc[-1]['us30_open'],
                    'us30_high': self.df.iloc[-1]['us30_high'],
                    'us30_low': self.df.iloc[-1]['us30_low'],
                    'us30_change':self.df.iloc[-1]['us30_change']}

                
                df4 = pd.DataFrame(data_z)
                df4['Date'] = pd.to_datetime(df4['Date'])
                df4.set_index('Date', inplace=True)

                return df4
            else:


                us30_data = investpy.get_index_historical_data(index='Dow 30',
                                                    country='united states',
                                                    from_date=from_day,
                                                    to_date=to_day)

                us30df = pd.DataFrame(us30_data)
                us30df = us30df.drop(columns=['Volume','Currency'])
                us30df = us30df.rename(columns={'Open':'us30_open','High':'us30_high','Low':'us30_low','Close':'us30_price'})
                df4 = pd.DataFrame(us30df, columns=['us30_price', 'us30_open', 'us30_high', 'us30_low'])
                df4.ffill()
                df4.bfill()
                df4['us30_change']=round(((df4['us30_price']-df4['us30_open'])/df4['us30_open'])*100, 2)

                return df4

    def get_usdidxdata(self):
        day = date.today() - timedelta(days=1)
        tday = date.today() + timedelta(days=1)
        data_lastday = datetime(*self.latest_date_tuple).date()
        last_day = datetime(*self.latest_date_tuple).date() + timedelta(days=1)

        if data_lastday < day:
            from_day = last_day.strftime("%d/%m/%Y")
            to_day = tday.strftime("%d/%m/%Y")

            d1 = last_day + timedelta(days=1)
            d2 = date.today()
            if d1 == d2 and self.is_weekend(d1)==1 and self.is_weekend(d2)==1:
                data_z = {
                    'Date': [last_day, d1],
                    'usidx_price': self.df.iloc[-1]['usidx_price'],
                    'usidx_open': self.df.iloc[-1]['usidx_open'],
                    'usidx_high': self.df.iloc[-1]['usidx_high'],
                    'usidx_low': self.df.iloc[-1]['usidx_low'],
                    'usidx_change':self.df.iloc[-1]['usidx_change']}

                
                df5 = pd.DataFrame(data_z)
                df5['Date'] = pd.to_datetime(df5['Date'])
                df5.set_index('Date', inplace=True)

                return df5
            else:


                usdidx_data = investpy.get_index_historical_data(index='US Dollar Index',
                                                    country='united states',
                                                    from_date=from_day,
                                                    to_date=to_day)

                usdidxdf = pd.DataFrame(usdidx_data)
                usdidxdf = usdidxdf.drop(columns=['Volume','Currency'])
                usdidxdf = usdidxdf.rename(columns={'Open':'usidx_open','High':'usidx_high','Low':'usidx_low','Close':'usidx_price'})
                df5 = pd.DataFrame(usdidxdf, columns=['usidx_price', 'usidx_open', 'usidx_high', 'usidx_low'])
                df5.ffill()
                df5.bfill()
                df5['usidx_change']=round(((df5['usidx_price']-df5['usidx_open'])/df5['usidx_open'])*100, 2)

                return df5


    def RSI(self, close, period=14):
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def update_data(self):
        
        if date.today() > self.latest_predicted_date_update:
            print(date.today())
            
            print('Data collection and transformation started...')
            print('Stage 1 ...')
            df1 = self.fetch_data(self.get_btcdata)
            print('Stage 2 ...')
            df2 = self.fetch_data(self.get_golddata)
            print('Stage 3 ...')
            df3 = self.fetch_data(self.get_spdata)
            print('Stage 4 ...')
            df4 = self.fetch_data(self.get_us30data)
            print('Stage 5 ...')
            df5 = self.fetch_data(self.get_usdidxdata)
            

            if all(df is not None for df in [df1, df2, df3, df4, df5]):
                print('Data collection process completed...')

                concatenated_df = pd.concat([df1,df2,df3,df4,df5],axis=1, join='outer')
                df_filled_forward = concatenated_df.ffill()
                df_filled = df_filled_forward.bfill()
                df_filled.index = df_filled.index.strftime('%Y-%m-%d')
                df_copy = self.df.copy()
                combined_df = pd.concat([df_copy, df_filled])
                combined_df['btc_rsi'] = self.RSI(combined_df['btc_price'])
                combined_df['btc_next'] = combined_df['btc_open'].shift(-1)
                to_predict = combined_df.iloc[-1][self.columns_of_interest].values
                new_df = combined_df.iloc[-10:, combined_df.columns.isin(self.columns_of_interest)]
                combined_df = combined_df.drop(combined_df.index[-1])
                combined_df['direction'] = combined_df.apply(lambda row: 1 if row['btc_next'] - row['btc_open'] > 0 else 0, axis=1)
                combined_df = combined_df.bfill()
                combined_df = combined_df.drop(columns='btc_next')
                last9_directions = combined_df['direction'].tail(10)
                direction_df = last9_directions.to_frame()
                new_df = pd.merge(new_df, direction_df, left_index=True, right_index=True, how='outer')

                combined_df.to_csv(self.data_file, index=True)
                new_df.to_csv(self.predict_file, index=True)

                #Bitcoin data last 10 days
                btc_data = new_df[['btc_price','btc_open','btc_high','btc_low','btc_change','btc_rsi']]
                #btc_data2 = btc_data.set_index('Date')
                btc_data.tail(10).to_csv(self.btc_file, index=True)

                last_9_rows_values = combined_df.iloc[-9:][self.columns_of_interest].values
                last_10_rows_values = np.vstack([last_9_rows_values,to_predict])
                last_9_test = combined_df.iloc[-9:]['direction'].values

                loaded_model = load_model(self.model_path)
                predictions = loaded_model.predict(last_10_rows_values)
                y_pred_binary = np.round(predictions).astype(int).flatten()

                direction = y_pred_binary[-1]
                print('Getting direction...')

                #updating predicted results
                newdff = pd.read_csv(self.predict_file)
                new_dff = pd.DataFrame({'Date': pd.to_datetime(newdff['Date'][-10:]),  'pred': y_pred_binary})
                existing_df = pd.read_csv(self.pred_results, index_col='Date', parse_dates=True)
                combined_dff = pd.concat([existing_df, new_dff.set_index('Date')])
                # Drop duplicate index rows, keeping only the first occurrence
                combined_dff = combined_dff[~combined_dff.index.duplicated(keep='first')]
                # Save the combined DataFrame back to the CSV file
                combined_dff.to_csv(self.pred_results)
                print('Updated predictions...')

                #accuracy
                last_preds = combined_dff['pred'][-10:-1].values
                accuracy_last_9 = accuracy_score(last_9_test, last_preds)*100
                accuracy_last_9 = np.round(accuracy_last_9, 1)
                print('Getting accuracy of last 10 days...')

                #for table
                dates = new_df.index[1:10].tolist()
                predicted = last_preds
                actual = last_9_test
                predicted_flat = predicted #.flatten()
                print('Loading table...')
                df_table = pd.DataFrame({'Date': dates, 'Prediction': predicted_flat, 'Actual': actual})
                # Map numeric values to corresponding categories
                df_table['Prediction'] = df_table['Prediction'].map({0: 'Fall', 1: 'Rise'})
                df_table['Actual'] = df_table['Actual'].map({0: 'Fall', 1: 'Rise'})


                acc = [0, accuracy_last_9]
                df_acc = pd.DataFrame(acc)
                df_acc.to_csv(self.latest_accuracy, index=True)

                dis_date = date.today()


                print('Data trnsformation process completed...')

                return direction, df_table, accuracy_last_9, dis_date
            
            else:
                print('Try again...')
        
        else:
            print('Using cache data')
            latest_data = pd.read_csv(self.predict_file)
            latest_data.set_index('Date', inplace=True)
            price_data = pd.read_csv(self.data_file)
            price_data.set_index('Date', inplace=True)
            pred_df = pd.read_csv(self.pred_results)
            pred_df.set_index('Date', inplace=True)



            last_preds = pred_df['pred'][-10:-1].values
            last_9_test = price_data.iloc[-9:]['direction'].values
            accuracy_last_9 = accuracy_score(last_9_test, last_preds)*100
            accuracy_last_9 = np.round(accuracy_last_9, 1)
            direction = pred_df['pred'][-1]

            #for table
            dates = latest_data.index[1:10].tolist()
            predicted = pred_df['pred'][-10:-1].values
            actual = last_9_test
            predicted_flat = predicted #.flatten()
            df_table = pd.DataFrame({'Date': dates, 'Prediction': predicted_flat, 'Actual': actual})
            # Map numeric values to corresponding categories
            df_table['Prediction'] = df_table['Prediction'].map({0: 'Fall', 1: 'Rise'})
            df_table['Actual'] = df_table['Actual'].map({0: 'Fall', 1: 'Rise'})

            acc = [0, accuracy_last_9]
            df_acc = pd.DataFrame(acc)
            df_acc.to_csv(self.latest_accuracy, index=True)

            dis_date = date.today()

            return direction, df_table, accuracy_last_9, dis_date
        
    
    def get_latest_accuracy(self):

        df = pd.read_csv(self.latest_accuracy)
        accuracy_updated = int(df.iloc[-1][-1])

        return accuracy_updated
    
    def get_btc_price(self):
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            btc_price = data["bitcoin"]["usd"]
            return btc_price
        else:
            print("Failed to fetch data")
            return None
        




        




 

    
