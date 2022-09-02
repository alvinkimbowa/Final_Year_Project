# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:30:32 2022

@author: ALVIN
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import *
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

random.seed(125)

#%%
def pv_pred():
    folder = 'deployment_original/streamlit/fyp/Data'
    file = "Site_3_01-2018_02-2018.csv"
            
    df = pd.read_csv(os.path.join(folder, file))
    df['Pyranometer_1'] = df['Pyranometer_1'].apply(lambda x: x if x<2000 and x>0 else 0)
    df['Global_Horizontal_Radiation'] = df['Global_Horizontal_Radiation'].apply(lambda x: x if x<2000 and x>0 else 0)
    df['Active_Power'] = df['Active_Power'].apply(lambda x: x if x>0 else 0)
    df.Timestamp = df.Timestamp.apply(lambda x: datetime.strptime(x, '%d-%m-%y %H:%M'))
    # print(df.head())
    

    train_start_date = datetime(2018,1,1,0)
    train_end_date = datetime(2018,1,24,23)
    train_df = df[df.Timestamp >= train_start_date].reset_index(drop=True)
    train_df = train_df[train_df.Timestamp <= train_end_date].reset_index(drop=True)
    
    val_start_date = datetime(2018,1,25,0)
    val_end_date = datetime(2018,1,28,23)
    val_df = df[df.Timestamp >= val_start_date].reset_index(drop=True)
    val_df = val_df[val_df.Timestamp <= val_end_date].reset_index(drop=True)
    
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
    
    train_data = get_variables(train_df, variables)
    val_data = get_variables(val_df, variables)
    test_data = get_variables(test_df, variables)
    
    
    feature_cols = list(set(train_data.columns) - set(target))
    # print("Feature columns: ", feature_cols)
    
    train_data_scaled = train_data.copy()
    val_data_scaled = val_data.copy()
    test_data_scaled = test_data.copy()
    
    sc_X = MinMaxScaler()
    train_data_scaled[feature_cols] = sc_X.fit_transform(train_data[feature_cols])
    val_data_scaled[feature_cols] = sc_X.transform(val_data[feature_cols])
    test_data_scaled[feature_cols] = sc_X.transform(test_data[feature_cols])
    
    sc_y = MinMaxScaler()
    train_data_scaled[target] = sc_y.fit_transform(train_data[target])
    val_data_scaled[target] = sc_y.transform(val_data[target])
    test_data_scaled[target] = sc_y.transform(test_data[target])
    
    
    models = {
        'Baseline': 0,
        'next hour seq_2_seq': 1,
        'next hour power pred': 2
        }
    
    # model = models['Baseline']
    model = models['next hour seq_2_seq']
    # model = models['next hour power pred']
    
    # This will ensure that you always stop to check whether you're
    # using the correct model
    # assert False
    
    
    n_in = 23
    n_out = 2
    
    X_train, y_train = get_supervised_data(train_data_scaled, n_in, n_out, model=model, add_dim=True, dropnan=True)
    X_val, y_val = get_supervised_data(val_data_scaled, n_in, n_out, model=model, add_dim=True, dropnan=True)
    X_test, y_test = get_supervised_data(test_data_scaled, n_in, n_out, model=model, add_dim=True, dropnan=True)
    
    
    # print("Features: ", X_train.shape, X_val.shape, X_test.shape)
    # print("Targets : ", y_train.shape, y_val.shape, y_test.shape)
    
    # It's always safe to check if the features and target dimensions are correct
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1:] == X_val.shape[1:] == X_test.shape[1:]
    assert y_train.shape[1:] == y_val.shape[1:] == y_test.shape[1:]
    

    model_path = "deployment_original/streamlit/fyp/models/pv_model.h5"
    
    model = tf.keras.models.load_model(model_path)
    # print(model.summary())
    
    model.evaluate(X_test, y_test)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    
    y_pred = model.predict(X_test)
    # y_pred = y_pred[:,-1]
    y_true = y_test.copy()
    
    
    # print("y_test: ", y_test.shape)
    print(y_pred.shape, y_true.shape)
    
    nrmse = np.sqrt(model.evaluate(X_test, y_test, verbose=0))
    
    y_pred = sc_y.inverse_transform(y_pred.reshape(-1,1))
    y_true = sc_y.inverse_transform(y_true.reshape(-1,1))
    
    y_pred = y_pred.reshape(-1,24,1)
    y_true = y_true.reshape(-1,24,1)
    
    y_pred = y_pred[:,-1,0]
    y_true = y_true[:,-1,0]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print("nRMSE: {}(kW)".format(nrmse))
    print("RMSE: {}(kW)".format(rmse))
    print("MAE: {}(kW)".format(mae))
    # print("MAPE: ", mape)
    
    df = pd.DataFrame({'Actual Active Power':y_true, 'Prediction':y_pred})
    
    return df
#%%    
# results = pv_pred()

# print(results.shape)
# plt.plot(results)