import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime

from bs4 import BeautifulSoup
import re
import os
import glob
import requests
from urllib.request import urlopen
import holidays

from prophet import Prophet
from prophet.serialize import model_from_json

import ephem

def data_loader(price:str, days:int):
    '''
    download electricity price data from omie data with the followin params
    price: day_ahead or intraday_price)
    days: last number of days of desired prices
    '''
    omie_link = 'https://www.omie.es/'
    
    if price == 'day-ahead':
        weblink = "https://www.omie.es/en/file-access-list?parents%5B0%5D=/&parents%5B1%5D=Day-ahead%20Market&parents%5B2%5D=1.%20Prices&dir=%20Day-ahead%20market%20hourly%20prices%20in%20Spain&realdir=marginalpdbc"
        col_list = [0,1,2,3,4] 
        skip = 1
        
    elif price == 'intraday':
        weblink = "https://www.omie.es/en/file-access-list?parents%5B0%5D=/&parents%5B1%5D=Continuous%20Intraday%20Market&parents%5B2%5D=1.%20Prices&dir=Maximum%2C%20minimum%20and%20weighted%20price%20for%20each%20hour%20of%20the%20continuous%20intraday%20market&realdir=precios_pibcic"
        col_list = [0,1,2,3,10] 
        skip = 3

    working_dir = os.getcwd()
    local_dir = os.mkdir(os.path.join(working_dir, price))
    os.chdir(price)
    
    #loading files list
    files_list = []
    soup = BeautifulSoup(urlopen(weblink))

    for link in soup.findAll('a'):
        if link.get('href').endswith('.1') or link.get('href').endswith('.2'):
            files_list.append(link.get('href'))
            
    #download files
    files_list = files_list[:days]
    for i in range(len(files_list)):
        r = requests.get(omie_link+files_list[i])
        with open(files_list[i][-10:-2]+'.txt', 'wb') as f: 
            f.write(r.content)
        
    #format to pandas dataframe
    data = pd.DataFrame(columns=['timestamp', price])
    for file in glob.glob('*.txt'):
        try:
          partial_data = pd.read_csv(file,
                                    sep=';',
                                    header=None,
                                    usecols = col_list,
                                    names = ['year', 'month', 'day', 'hour', price],
                                    skiprows=skip,
                                    skipfooter=1,
                                    dtype={'year':int, 'month':int, 'day':int, 'hour':int},
                                    encoding='latin-1',
                                    engine='python'
                                    )
          partial_data['hour'] = partial_data['hour'].replace(24, 0)
          partial_data.drop(partial_data[partial_data.hour > 23].index, inplace=True)

          for i in range(len(partial_data)):
            partial_data.loc[i, 'date'] = str(partial_data.loc[i, 'year'])+'/'+str(partial_data.loc[i, 'month'])+'/'+str(partial_data.loc[i, 'day'])+':'+str(partial_data.loc[i, 'hour'])
            s = str(partial_data.loc[i, 'date'])
            partial_data.loc[i, 'timestamp'] = datetime.datetime.strptime(s,"%Y/%m/%d:%H").timestamp()

          data = pd.concat([data, partial_data[['timestamp', price]]], axis=0).sort_values(by=['timestamp'])
          del partial_data
        except:
          continue
     
    os.chdir(working_dir)
    return data

def signal(x):
    '''
    return 1 for profit and zero for non-profit or loss
    '''
    return 1 if x > 0 else 0

def day_night(lat='40.41669090', long='-3.70034540', elevation=653, timestamp=0):
  '''
  determine wether it is day or night in madrid according to sun location in
  provided timestamp

  return:
    1 : day
    0 : night
  '''

  sun = ephem.Sun()
  observer = ephem.Observer()
  # coordinates for Madrid
  observer.lat, observer.lon, observer.elevation = lat, long, elevation
  # Set the time for Madrid
  observer.date = datetime.datetime.strptime(str(np.datetime64(timestamp, 's')), '%Y-%m-%dT%H:%M:%S')
  sun.compute(observer)
  current_sun_alt = sun.alt

  return 1 if (current_sun_alt*180/math.pi > -6) else 0

 def data_formater(dap, idp): 
    
    # merging two dataframes into one
    data = pd.merge(dap, idp, on='timestamp', how='left').dropna().reset_index(drop=1)

    # intraday price format
    data['intraday'] = data['intraday'].str.replace(',', '.').astype(float)

    # compute the spread 
    data['spread'] = data['intraday'] - data['day-ahead']

    # compute gain/loss signal
    data['signal'] = data['spread'].apply(signal)

    # adding a date columns for readability and easier dummy variable extraction
    data['date'] = data['timestamp'].apply(lambda x: pd.to_datetime(x, unit='s', errors='coerce'))

    # getting additionnal categorical data for
    ## day of the week with monday = 0 and sunday = 6
    data['day_of_week'] = data['date'].dt.dayofweek

    ## determine whether it is weekend days
    data['is_weekend'] = (data['date'].dt.dayofweek > 4).astype(int)

    ## week number
    data['week_number'] = data['date'].dt.isocalendar().week.astype(int)

    ## season of the year
    data['season'] = data['date'].dt.month%12 // 3 + 1

    # adding spanish holidays
    for i in range(len(data)):
      data.loc[i, 'is_holiday'] = (data.loc[i, 'date'] in holidays.country_holidays('ES'))

    data['is_holiday'] = data['is_holiday'].astype(int)

    # Adding whether it is day or night
    for i in range(len(data)):
      data.loc[i, 'day_night'] = day_night(timestamp = data.loc[i, 'date'])

    data['day_night'] = data['day_night'].astype(int)

    # change cols names to model settings
    data.rename(columns={'date':'ds', 'spread':'y'}, inplace=True)

    # adding lagged cols
    for lag in range(1,25):
        data['y_lag_' + str(lag)] = data.y.shift(lag)

    #  rolling mean
    data['rolling_mean_24h'] = data['y'].rolling(24).mean()    

    # dropna
    data.dropna(inplace=True)
    
    # reset index
    data.reset_index(drop=True, inplace=True)
    
    return data

# future dataframe
def future_to_model(data_to_model):
    '''
    preprocess next 24h to apply model
    '''
    futures = pd.DataFrame(pd.date_range(data_to_model.iloc[-1]['ds'], 
                                         freq="h",
                                         periods=25, 
                                         inclusive='right'),
                           columns=['ds'])
    
     # getting additionnal categorical data for
    ## day of the week with monday = 0 and sunday = 6
    futures['day_of_week'] = futures['ds'].dt.dayofweek

    ## determine whether it is weekend days
    futures['is_weekend'] = (futures['ds'].dt.dayofweek > 4).astype(int)

    ## week number
    futures['week_number'] = futures['ds'].dt.isocalendar().week.astype(int)

    ## season of the year
    futures['season'] = futures['ds'].dt.month%12 // 3 + 1

    # adding spanish holidays
    for i in range(len(futures)):
      futures.loc[i, 'is_holiday'] = (futures.loc[i, 'ds'] in holidays.country_holidays('ES'))

    futures['is_holiday'] = futures['is_holiday'].astype(int)

    # Adding whether it is day or night
    for i in range(len(futures)):
      futures.loc[i, 'day_night'] = day_night(timestamp = futures.loc[i, 'ds'])

    futures['day_night'] = futures['day_night'].astype(int)
    
    return futures





