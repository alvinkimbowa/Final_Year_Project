# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:27:43 2022

@author: ALVIN
"""

import numpy as np
import pandas as pd
from datetime import datetime


def adjust_day(day):
    '''This function changes the day of the week such that 1 means Sunday and 
        7 means Saturday

    The input 'day' should be in the range [0-6] where 0 means Monday and 6 
    means Sunday
    '''
    if day == 6:
        day = 1
    else:
        day += 2

    return day


def get_schedule_variables(timestamp):
    '''This function extracts the hour of the day, day of the week, day of the
        month, and the month of the year.
        
        The date_time_str argument should have the format '2016-18-02 08:00:00'

        Ranges of the variables
            hr - (0-23)
            day_of_week - (0 - Monday, 6 - Sunday)
            day_of_month - (1-31)
            mnth - (1-12)
    '''
    if not isinstance(timestamp, datetime):
        timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    
    # print ("The type of the date is now",  type(timestamp))
    # print ("The date is", timestamp)
    hr = timestamp.hour
    day_of_week = adjust_day(timestamp.weekday())
    day_of_month = timestamp.day
    mnth = timestamp.month

    return [hr, day_of_week, day_of_month, mnth]


def get_variables(data, variables, schedule_variables=True):
    '''This function extracts the required weather variables and also includes
        schedule variables as a default.
    
    Arguments:
        data - a dataFrame and should contain the required variables
        varialbes - a list of strings of the columns required   
    '''

    if schedule_variables:
        data['sch_variables'] = data['Timestamp'].apply(get_schedule_variables)
        data['hr'] = data['sch_variables'].apply(lambda x: x[0])
        data['day_of_week'] = data['sch_variables'].apply(lambda x: x[1])
        data['day_of_month'] = data['sch_variables'].apply(lambda x: x[2])
        data['Month'] = data['sch_variables'].apply(lambda x: x[3])

        variables = ['hr', 'day_of_week', 'day_of_month', 'Month'] + variables
        
    # Calcuate the diffuse horizontal radiation
    data['Diffuse_Horizontal_Radiation'] = data['Pyranometer_1'] - data['Global_Horizontal_Radiation']
    data = data[variables]
    
    return data


# This function was obtained from
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def get_supervised_data(variables, n_in=23, n_out=2, add_dim=False, dropnan=True,
                        verify=False, model=0):
    no_var = len(variables.columns)
    variables = variables.values
    # Save a copy of the original data for purposes of data verification
    orig_data_0 = series_to_supervised(variables, n_in, n_out, dropnan)
    orig_data = orig_data_0.values

    # orig_data = orig_data[:,:-no_var]
    data = orig_data.reshape((-1,int(orig_data.shape[1]/no_var),no_var))
    if verify:
        print("Original data shape:\t", orig_data_0.shape)
        print("Supervised data shape: ", data.shape)
    

    if verify:
        return data, orig_data_0
    else:
        # print("no_var: ", no_var)
        # print("data: ", data.shape)
        
        if model==0:
            # Baseline
            X = data[:,:24,:no_var-1]
            y = data[:,:24,no_var-1]
            y = np.expand_dims(y, axis=2)
        
        elif model==1:    
            # next hour seq_2_seq
            X = data[:,:24,:no_var-1]
            y = data[:,1:,-1]
            y = np.expand_dims(y, axis=2)

        elif model==2:
            # next hour power pred
            X = data[:,:24,:no_var-1]
            y = data[:,-1,-1]

        else:
            print("Please select a model")
            return None

        return (X,y)
    
    

def read_pv_data(filename):
    df = pd.read_csv(filename)
    df['Pyranometer_1'] = df['Pyranometer_1'].apply(lambda x: x if x<2000 and x>0 else 0)
    df['Global_Horizontal_Radiation'] = df['Global_Horizontal_Radiation'].apply(lambda x: x if x<2000 and x>0 else 0)
    df['Active_Power'] = df['Active_Power'].apply(lambda x: x if x>0 else 0)
    df.Timestamp = df.Timestamp.apply(lambda x: datetime.strptime(x, '%d-%m-%y %H:%M'))

    test_start_date = datetime(2018,1,29,0)
    test_end_date = datetime(2018,2,1,23)
    test_df = df[df.Timestamp >= test_start_date].reset_index(drop=True)
    test_df = test_df[test_df.Timestamp <= test_end_date].reset_index(drop=True)

    features = [
            'Global_Horizontal_Radiation',
            'Pyranometer_1',
            ]

    target = ['Active_Power']

    variables = features + target

    return get_variables(test_df, variables)


def read_load_data(filename):
    # df = pd.read_csv("Data/Residential_1.csv")
    df = pd.read_csv(filename)
    df['year'] = df['date'].apply(lambda x: x[:4])
    df['month'] = df['date'].apply(lambda x: x[5:7])
    df['day'] = df['date'].apply(lambda x: x[-2:])
    # df = pd.DataFrame(data=df, index=None)
    # col_names = df.columns.tolist()
    # df = df[['year', 'month', 'day', 'hour', 'energy_kWh']].values.tolist()
    df = df[['year', 'month', 'day', 'hour', 'energy_kWh']]
    # # print(col_names)

    return df
