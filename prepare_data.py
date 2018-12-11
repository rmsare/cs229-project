""" Prepare air quality and data for single city """

import os
import numpy as np
import pandas as pd


def assign_class(df, col, threshs):
    df['class'] = np.zeros(len(df))
    for i, thresh in enumerate(threshs):
        high = (df[col] >= thresh)
        df['class'].loc[high] = i + 1
    return df


def lag(df, column, lags=[1]):
    for lag in lags:
        df[column + '_' + str(lag)] = df[column].shift(lag)
    return df


def load_data(city):
    df1 = load_pollutants(city)
    df2 = load_weather(city)
    df3 = load_pm25(city)
    df = pd.concat([df1, df2, df3], axis=1)
    df = df.dropna(axis='index')
    return df


def load_pm25(city):
    filename = 'data/' + city.lower().replace(' ', '-') + '-pm25.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = df.set_index(pd.to_datetime(df['Date']))
    else:
        df = pd.DataFrame([])
    return df


def load_pollutants(city):
    df = pd.read_csv('data/pollution_us_2000_2016.csv')
    df = df[df.City == city]
    df = df[['Date Local','NO2 AQI','O3 AQI','SO2 AQI','CO AQI']]
    df = df.dropna(axis='index')
    df = df.set_index(pd.to_datetime(df['Date Local'], format='%Y-%m-%d'))
    df = df.groupby(pd.TimeGrouper('D')).mean()
    return df


def load_precipitation(city):
    filename = 'data/' + city.lower().replace(' ', '-') + '-precip.csv'
    df = pd.read_csv(filename)
    return df


def load_weather(city):
    files = ['data/temperature.csv', 'data/humidity.csv',
            'data/wind_speed.csv', 'data/wind_direction.csv']
    names = ['temp', 'hum', 'wind_spd', 'wind_dir']
    dfs = []
    for name, f in zip(names, files):
        temp = pd.read_csv(f)[['datetime', city]]
        temp.columns = ['datetime', name]
        temp['datetime'] = pd.to_datetime(temp['datetime'])
        temp = temp.set_index('datetime')
        dfs.append(temp)
    df = pd.concat(dfs, axis=1)

    df_mean = df.groupby(pd.TimeGrouper('D')).mean()
    df_mean.columns = ['temp_mean', 'hum_mean', 'ws_mean', 'wd_mean']
    df_min = df.groupby(pd.TimeGrouper('D')).min()[['temp']]
    df_min.columns = ['temp_min']
    df_max = df.groupby(pd.TimeGrouper('D')).max()[['temp', 'hum', 'wind_spd']]
    df_max.columns = ['temp_max', 'hum_max', 'ws_max']
    df = pd.concat([df_mean, df_min, df_max], axis=1)
    return df
