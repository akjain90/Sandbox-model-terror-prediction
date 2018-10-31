# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:46:11 2018

@author: jain
"""
from modules.prepare_data import prepare_data
import numpy as np
import matplotlib.pyplot as plt

#def main():
data_name = 'sandbox_attacks'
data_dir = '../../'+data_name+'.csv'
train_data,train_date,test_data,test_date,std = prepare_data(data_dir,test_len=30,normalize=False)
weight_aray = np.linspace(0,1,100)
feed = train_data[-100:].flatten()
pred = np.divide(np.sum(np.multiply(weight_aray,feed)),np.sum(weight_aray))
pred_arr = np.ones((len(test_data)))*pred

plt.figure()
plt.plot_date(test_date,test_data,xdate=True,label='Labels',ls="-")
plt.plot_date(test_date,pred_arr,xdate=True,label='Predictions',ls="-")
plt.xticks(rotation="vertical")
plt.title('Prediction')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Attack')
#save_fig(data_name,'./Images/')
    