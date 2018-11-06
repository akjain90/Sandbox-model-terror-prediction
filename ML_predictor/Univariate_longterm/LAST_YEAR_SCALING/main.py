# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:55:44 2018

@author: jain
"""
import matplotlib.pyplot as plt
from modules.prepare_data import prepare_data
from modules.training import training
from modules.predict import predict
from modules.save_fig import save_fig
import pandas as pd
import numpy as np
from modules.differential_loss import differential_loss
from modules.normed_loss import normed_loss

def main():
    loss_norm = []
    loss_difference = []
    countries = ['Afghanistan','Indien','Irak','Kolumbien','Pakistan','Philippinen','sandbox_attacks','test_exp_chirp']
    for country in countries:
        test_len = 30
        data_name = country
        data_dir = '../../'+data_name+'.csv'
        train_data,train_date,test_data,test_date,std = prepare_data(data_dir,test_len=test_len,normalize=False)
        
        scale_down_mean = np.mean(train_data[-365-365+test_len:-365])
        scale_down_std = np.std(train_data[-365-365+test_len:-365])
        scale_up_mean = np.mean(train_data[-365+30:])
        scale_up_std = np.std(train_data[-365+30:])
        
        prediction = ((train_data[-365:-365+30]-scale_down_mean)/scale_down_std)*scale_up_std+scale_up_mean
        
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
    pd.DataFrame(loss_dict).to_csv('./Last_year_scaling_loss.csv')

if __name__=='__main__':
    main()