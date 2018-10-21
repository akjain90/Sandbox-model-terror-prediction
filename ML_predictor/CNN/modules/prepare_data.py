# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:52:47 2018

@author: akjai
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def prepare_data(file,train=60,test=20,cv=20,normalize=False):
    data = pd.read_csv(file,index_col=0)

    data_val = data.values
    
    train_len = np.floor_divide(train*len(data_val),100)
    test_len = np.floor_divide(test*len(data_val),100)
    
    train_set = data_val[:train_len,1:]
    train_date = data_val[:train_len,0]
    
    test_set = data_val[train_len:train_len+test_len,1:]
    test_date = data_val[train_len:train_len+test_len,0]
    
    cv_set = data_val[train_len+test_len:,1:]
    cv_date = data_val[train_len+test_len:,0] 
    std = StandardScaler()
    
    if normalize==False:
        return train_set,train_date,test_set,test_date,cv_set,cv_date,std
    else: 
        train_std = std.fit_transform(train_set)
        test_std = std.transform(test_set)
        cv_std = std.transform(cv_set)
        return train_std,train_date,test_std,test_date,cv_std,cv_date,std
    