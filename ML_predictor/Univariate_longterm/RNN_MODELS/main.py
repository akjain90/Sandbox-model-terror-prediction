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

def main():
    data_name = 'test_exp_chirp'
    data_dir = '../../'+data_name+'.csv'
    train_data,train_date,test_data,test_date,std = prepare_data(data_dir,normalize=True,scaling='minmax')  
    model_dir = './model/'
    num_epoch = 1000
    n_steps = 100
    n_inputs = 1
    sess, train_loss,epoch_count = training(train_data,model_dir,num_epoch=num_epoch,n_steps=n_steps,n_inputs=n_inputs)
        
    prediction,true_labels,label_dates = predict(test_data,test_date,sess,model_dir,n_steps,n_inputs)   
    rescaled_prediction = std.inverse_transform(prediction.reshape(-1,1))
    rescaled_labels = std.inverse_transform(true_labels.reshape(-1,1))
    
    plt.figure()
    plt.subplot(211)
    plt.plot(epoch_count,train_loss)
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training MSE')
    plt.subplot(212)
    plt.plot_date(label_dates,rescaled_labels,xdate=True,label='Labels',ls="-")
    plt.plot_date(label_dates,rescaled_prediction,xdate=True,label='Predictions',ls="-")
    plt.xticks(rotation="vertical")
    plt.title('Prediction')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Attack')
    save_fig('predicted value feedback'+data_name,'./Images/')

if __name__=='__main__':
    main()