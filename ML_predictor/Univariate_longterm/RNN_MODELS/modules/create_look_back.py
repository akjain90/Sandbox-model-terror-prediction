# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:25:12 2018

@author: jain
"""
import numpy as np
def create_look_back(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back:i + look_back+1, 0])
    return np.array(dataX), np.array(dataY)
