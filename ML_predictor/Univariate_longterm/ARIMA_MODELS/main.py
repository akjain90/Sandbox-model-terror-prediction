# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:55:44 2018

@author: jain
"""
import matplotlib.pyplot as plt
from modules.prepare_data import prepare_data
#from modules.training import training
#from modules.predict import predict
from modules.save_fig import save_fig
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from scipy.signal import correlate
import pandas as pd

#def main():
data_name = 'Philippinen'
data_dir = '../../'+data_name+'.csv'
train_data,train_date,test_data,test_date,std = prepare_data(data_dir,test_len=30,normalize=True,scaling='minmax')
autocorrelation_plot(train_data)
plt.show()
#history = train_data.flatten().tolist()
#model = ARIMA(history, order=(30,0,0))
#model_fit = model.fit(disp=0)
#output = model_fit.forecast(steps=30)
#
##plt.plot_date(test_date,test_data,xdate=True,label='Labels',ls="-")
#plt.plot_date(test_date,output[0],xdate=True,label='Predictions',ls="-")
#plt.xticks(rotation="vertical")
#plt.title('Prediction')
#plt.legend()
#plt.xlabel('Days')
#plt.ylabel('Attack')
#save_fig('predicted value feedback'+data_name,'./Images/')

#if __name__=='__main__':
#    main()