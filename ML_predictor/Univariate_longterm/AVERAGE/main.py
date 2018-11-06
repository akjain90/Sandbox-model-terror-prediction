# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:46:11 2018

@author: jain
"""
from modules.prepare_data import prepare_data
import numpy as np
import matplotlib.pyplot as plt
from modules.differential_loss import differential_loss
from modules.normed_loss import normed_loss
import pandas as pd
from modules.save_fig import save_fig

def main():
    loss_norm = []
    loss_difference = []
    countries = ['Afghanistan','Indien','Irak','Kolumbien','Pakistan','Philippinen','sandbox_attacks','test_exp_chirp']
    for country in countries:
        data_name = country
        data_dir = '../../'+data_name+'.csv'
        train_data,train_date,test_data,test_date,std = prepare_data(data_dir,test_len=30,normalize=False)
        weight_aray = np.linspace(0,1,100)
        feed = train_data[-100:].flatten()
        pred = np.divide(np.sum(np.multiply(weight_aray,feed)),np.sum(weight_aray))
        prediction = np.ones((len(test_data)))*pred
        
        loss_norm.append(normed_loss(prediction,test_data))
        loss_difference.append(differential_loss(prediction,test_data))
        
        plt.figure()
        plt.plot_date(test_date,test_data,xdate=True,label='Labels',ls="-")
        plt.plot_date(test_date,prediction,xdate=True,label='Predictions',ls="-")
        plt.xticks(rotation="vertical")
        plt.title('Prediction')
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Attack')
        save_fig(data_name,'./Images/')
    loss_dict = {'Countries':countries,'Normed_loss': loss_norm,'Differential_loss': loss_difference}
    pd.DataFrame(loss_dict).to_csv('./Average_loss.csv')


if __name__=='__main__':
    main()
    