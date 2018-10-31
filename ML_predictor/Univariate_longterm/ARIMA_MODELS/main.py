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


#def main():
data_name = 'test_exp_chirp'
data_dir = '../../'+data_name+'.csv'
train_data,train_date,test_data,test_date,std = prepare_data(data_dir,test_len=30,normalize=True,scaling='minmax')  
history = train_data.flatten().tolist()
model = ARIMA(history, order=(0,1,0))
model_fit = model.fit(disp=0)
output = model_fit.forecast(steps=30)
#model = ARIMA(history, order=(0,1,0))
#model_fit = model.fit(disp=0)
#prediction=[]
#for i in range(len(test_data)):
#    model = ARIMA(history, order=(550,1,0))
#    model_fit = model.fit(disp=0)
#    output = model_fit.forecast()
#    prediction.append(output[0][0])
#    # rolling prediction using last prediction
#    #history.append(output[0][0])
#    # rolling prediction using last true value
#    history.append(test_data[i,0])
#    print(i)
#    
#prediction = history[-30:]
#autocorrelation_plot(train_data)
#plt.show() 
    
plt.plot(output[0])
#plt.figure()
##plt.plot_date(test_date,test_data,xdate=True,label='Labels',ls="-")
#plt.plot_date(test_date,prediction,xdate=True,label='Predictions',ls="-")
#plt.xticks(rotation="vertical")
#plt.title('Prediction')
#plt.legend()
#plt.xlabel('Days')
#plt.ylabel('Attack')
#    save_fig('predicted value feedback'+data_name,'./Images/')
#
#if __name__=='__main__':
#    main()