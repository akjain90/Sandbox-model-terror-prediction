# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 22:50:54 2018

@author: akjai
"""
import numpy as np
def seq_fetch_batch(data, date, batch_size, l, w, pred_window,index,seq_window):
    num_steps = l*w
    data_len, features = data.shape
    X = []
    y = []
    y_date = []
    terminate = False
    if index*seq_window>(data_len-num_steps-pred_window-(batch_size-1)*seq_window):
        batch_size = np.floor_divide(data_len-index*seq_window-num_steps-pred_window,seq_window)+1
        terminate = True
    for i in range(batch_size):
        start = index*seq_window
        end = index*seq_window+num_steps
        temp_X = data[start:end,:].reshape(l,w,features)
        temp_y = data[end:end+pred_window,0].reshape(-1)
        # date only corrosponds to the labels that is y
        temp_date = date[end:end+pred_window]
        X.append(temp_X)
        y.append(temp_y)
        y_date.append(temp_date)
        index = index+1
    return np.array(X), np.array(y), np.array(y_date),index,terminate