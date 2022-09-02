# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:30:41 2022

@author: ALVIN
"""
import os
import glob
import keras
import random
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pprint import pprint
from natsort import natsorted
from datetime import datetime
from keras.models import load_model
# from IPython.display import clear_output
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

random.seed(125)

#%%
# Data Preprocessing

folder = 'deployment_original/streamlit/fyp/Data'
file = 'Residential_1.csv'

# pprint(files[:4])

load_df = pd.read_csv(os.path.join(folder, file))

df = pd.concat([load_df], axis=1)
df.date = df.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
# print(type(df.date[0]))
# print(df.head(5))

#%%
# Reduce outliers to maximum value
outliers = df[df['energy_kWh']>7]
df['energy_kWh'] = np.clip(df['energy_kWh'], a_max=6.5, a_min=min(df['energy_kWh']))


#%%
"""## Split data
Since the data is a lot, we can do an 80/10/10 split.

"""

# train_df = df.loc[:int(0.8*len(df))-1]
# val_df = df.loc[int(0.8*len(df)):int(0.9*len(df))-1]
# test_df = df.loc[int(0.9*len(df)):]
test_df = df.copy()

# test_df.to_csv('Residential_1.csv', index=False)


#%%
"""## Extracting out the variables
This includes the schedule variables and the weather variables.
"""
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

# adjust_day(5)

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

get_schedule_variables(df.date[13])

def get_variables(data, variables, schedule_variables=True):
    '''This function coverts the data to a supervised time series dataset.
    
    Arguments:
        data - a dataFrame and should contain the required variable(s)
        varialbes - a list of strings of the columns required   
    '''

    if schedule_variables:
        data['sch_variables'] = data['Timestamp'].apply(get_schedule_variables)
        data['hr'] = data['sch_variables'].apply(lambda x: x[0])
        data['day_of_week'] = data['sch_variables'].apply(lambda x: x[1])
        data['day_of_month'] = data['sch_variables'].apply(lambda x: x[2])
        data['Month'] = data['sch_variables'].apply(lambda x: x[3])

        variables = ['hr', 'day_of_week', 'day_of_month', 'Month'] + variables
        
    
    data = data[variables]
    
    return data

features = [
            # 'hour',
            'energy_kWh',
            ]

# target = ['Active_Power']
# variables = features + target
variables = features

# train_data = get_variables(train_df, variables, schedule_variables=False)
# val_data = get_variables(val_df, variables, schedule_variables=False)
test_data = get_variables(test_df, variables, schedule_variables=False)

#%%
"""## Creating time series data
This involves reshaping the data to have time steps.
"""

# This function was obtained from
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=2, dropnan=True):
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
	return agg.reset_index(drop=True)

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

time_steps = 23

# train = series_to_supervised(train_data, n_in=time_steps)
# val = series_to_supervised(val_data, n_in=time_steps)
test = series_to_supervised(test_data, n_in=time_steps)

# X_train = train.values[:,:-1]
# X_val = val.values[:,:-1]
X_test = test.values[:,:-1]

# y_train = train.values[:,-1:]
# y_val = val.values[:,-1:]
y_test = test.values[:,-1:]

# print("Features: ", X_train.shape, X_val.shape, X_test.shape)
# print("Targets : ", y_train.shape, y_val.shape, y_test.shape)

# # It's always safe to check if the features and target dimensions are correct
# assert len(X_train) == len(y_train)
# assert len(X_val) == len(y_val)
# assert len(X_test) == len(y_test)
# assert X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:]
# assert y_train.shape[1:] == y_val.shape[1:] == y_test.shape[1:]

#%%
"""## Feature scaling
Applying min-max normalization only the train and validation data. We don't need to do it on the test data.

"""
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler()
# X_train = sc_X.fit_transform(X_train)
# X_val = sc_X.fit_transform(X_val)
X_test = sc_X.fit_transform(X_test)

sc_y = MinMaxScaler()
# y_train = sc_y.fit_transform(y_train)
# y_val = sc_y.fit_transform(y_val)
y_test = sc_y.fit_transform(y_test)

# # Creating timesteps with 1 feature each
# X_train = np.expand_dims(X_train, axis=2)
# X_val = np.expand_dims(X_val, axis=2)
X_test = np.expand_dims(X_test, axis=2)

#%%
"""# Evaluation using best model
We use the best model to do the evaluation on all the datasets.
"""

model_path = 'deployment_original/streamlit/fyp/models/load_model.h5'
best_model = tf.keras.models.load_model(model_path)

#%%
"""### Quantitative Evaluation"""

def load_pred():
    y_pred = best_model.predict(X_test)
    # y_pred = y_pred[:,-1]
    y_true = y_test.copy()
    # y_true = y_true[:,-1]
    
    print(y_pred.shape, y_true.shape)
    
    y_pred = sc_y.inverse_transform(y_pred.reshape(-1,1))
    y_true = sc_y.inverse_transform(y_true.reshape(-1,1))
    
    y_pred = y_pred[:,-1]
    y_true = y_true[:,-1]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print("RMSE: {}(kW)".format(rmse))
    print("MAE: {}(kW)".format(mae))
    # print("MAPE: ", mape)
    
    y_true = y_true[-72:]
    y_pred = y_pred[-72:]
    df = pd.DataFrame({'Actual Energy Consumption':y_true, 'Prediction': y_pred})
    
    return df