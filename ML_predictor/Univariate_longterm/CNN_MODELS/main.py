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
from modules.differential_loss import differential_loss
from modules.normed_loss import normed_loss
import pandas as pd

def main():
    loss_norm = []
    loss_difference = []
    countries = ['Afghanistan','Indien','Irak','Kolumbien','Pakistan','Philippinen','sandbox_attacks','test_exp_chirp']
    for country in countries:
        
        #data_name = 'test_exp_chirp'
        data_name = country
        data_dir = '../../'+data_name+'.csv'
        train_data_scaled,train_date,test_data,test_date,std = prepare_data(data_dir,normalize=True,scaling='minmax')
        model_dir = './model/'
        num_epoch = 2000
        print(country)
        sess, train_loss,epoch_count = training(train_data_scaled,train_date,model_dir,num_epoch=num_epoch)
        print()
        
        prediction,true_labels,label_dates = predict(test_data,test_date,sess,std,model_dir)
        
#        rescaled_prediction = std.inverse_transform(prediction.reshape(-1,1))
#        rescaled_labels = std.inverse_transform(true_labels.reshape(-1,1))
        
#        loss_norm.append(normed_loss(rescaled_prediction,rescaled_labels))
#        loss_difference.append(differential_loss(rescaled_prediction,rescaled_labels))
        loss_norm.append(normed_loss(prediction,true_labels))
        loss_difference.append(differential_loss(prediction,true_labels))
        
        plt.figure()
        plt.subplot(211)
        plt.plot(epoch_count,train_loss)
        plt.title('Training loss')
        plt.xlabel('Epoch')
        plt.ylabel('Training MSE')
        plt.subplot(212)
        plt.plot_date(label_dates,true_labels,xdate=True,label='Labels',ls="-")
        plt.plot_date(label_dates,prediction,xdate=True,label='Predictions',ls="-")
        plt.xticks(rotation="vertical")
        plt.title('Prediction')
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Attack')
        save_fig(data_name,'./Images/')
    loss_dict = {'Countries':countries,'Normed_loss': loss_norm,'Differential_loss': loss_difference}
    pd.DataFrame(loss_dict).to_csv('./CNN_loss.csv')

if __name__=='__main__':
    main()