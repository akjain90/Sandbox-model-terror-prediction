# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:52:47 2018

@author: akjai
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def prepare_data(file,test_len = 130,normalize=False,scaling='minmax'):
    data = pd.read_csv(file,index_col=0)

    data_val = data.values
    
    train_set = data_val[:-test_len,0].reshape(-1,1)
    train_date = data_val[:-test_len,1]
    
    test_set = data_val[-test_len:,0].reshape(-1,1)
    test_date = data_val[-test_len:,1]
    
    std = None
    
    if normalize==False:
        return train_set,train_date,test_set,test_date,std
    elif(scaling=='standardscaler'):
        std = StandardScaler()
        train_std = std.fit_transform(train_set)
        test_std = std.transform(test_set)
        return train_std,train_date,test_std,test_date,std
    elif(scaling=='minmax'):
        std = MinMaxScaler()
        train_std = std.fit_transform(train_set)
        test_std = std.transform(test_set)
        return train_std,train_date,test_std,test_date,std
    