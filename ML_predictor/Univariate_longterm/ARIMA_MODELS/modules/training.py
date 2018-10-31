# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:33:50 2018

@author: jain
"""
from statsmodels.tsa.arima_model import ARIMA

def training(data,p,d,q):
    model = ARIMA(data, order=(p,d,q)) 
    model_fit = model.fit(disp=1)
    print(model_fit.summary())
    output = model_fit.forcast()
    return output
        