import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
from modules.fetch_batch import fetch_batch
from modules.save_fig import save_fig
from modules.prepare_data import prepare_data
#%%

train_std,train_date,test_std,test_date,cv_std,cv_date,std = prepare_data("../1_complex.csv",
                                                                      normalize=True)
#%%
# graph definition
tf.reset_default_graph()

l = 10
w = 10
#c = 3
c = 1
pred_window = 30
num_epoch = 2000
batch_size = 200
directory = '../saved_model/1_complex/only_attack/'


X = tf.placeholder(dtype = tf.float32, 
                   shape = (None, l,w,c), name="X")
y = tf.placeholder(dtype = tf.float32,
                   shape = (None,pred_window), name="y")

conv_1 = tf.layers.conv2d(X,
                          filters=4,
                          kernel_size=(5,5),
                          strides=1,
                          padding="same",
                          activation=tf.nn.relu, 
                          name = "conv_1")

#pool_1 = tf.layers.max_pooling2d(conv_1, 
#                                 pool_size=(3,3),
#                                 strides=2,
#                                 name = "max_pool_1")

conv_2 = tf.layers.conv2d(conv_1,
                          filters=8,
                          kernel_size=(5,5),
                          strides=1,
                          padding="same",
                          activation=tf.nn.relu, 
                          name = "conv_2")

pool_2 = tf.layers.max_pooling2d(conv_2, 
                                 pool_size=(3,3),
                                 strides=2,
                                 name = "max_pool_2")

#conv_3 = tf.layers.conv2d(pool_2,
#                          filters=64,
#                          kernel_size=(2,2),
#                          strides=1,
#                          padding="same",
#                          activation=tf.nn.relu, 
#                          name = "conv_3")
#
#pool_3 = tf.layers.max_pooling2d(conv_3, 
#                                 pool_size=(2,2),
#                                 strides=2,
#                                 name = "max_pool_3")
#
flatten = tf.layers.flatten(pool_2)
print(flatten)
dense_1 = tf.layers.dense(flatten, units=100, activation=tf.nn.relu, name="dense_1")

dense_2 = tf.layers.dense(dense_1, units=50, activation=tf.nn.relu, name="dense_2")

output = tf.layers.dense(dense_2, units=pred_window, name="output")

mse = tf.reduce_mean(tf.square(output-y),name="MSE")

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

training_op = optimizer.minimize(mse)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

#%%
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        X_batch, y_batch = fetch_batch(train_std, batch_size, l, w, pred_window)
        #X_batch, y_batch = fetch_batch(train_set, batch_size, l, w, pred_window)
        sess.run(training_op, feed_dict = {X:X_batch[:,:,:,2:], y: y_batch})
        if epoch%50==0:
            train_error = sess.run(mse, feed_dict = {X:X_batch[:,:,:,2:], y: y_batch})
            test_x, test_y = fetch_batch(test_std, 1, l, w, pred_window)
            #test_x, test_y = fetch_batch(test_set, 1, l, w, pred_window)
            test_error = sess.run(mse, feed_dict = {X:test_x[:,:,:,2:], y: test_y})
            print("Epoch: ",epoch, " Training error: ", train_error, " Test error: ", test_error)
    saver.save(sess,directory)
    #plt.figure()
    #plt.plot()
    

#print(mse)
    
#%%
#X_check, y_check = fetch_batch(test_set, 1, l, w, pred_window)
data_check = pd.read_csv("../random_new.csv",index_col=0)

data_check_val = data_check.values
print(data_check_val.shape)
random_check_set = data_check_val[:,1:]
random_check_dates = data_check_val[:,0]

print(random_check_set.shape)
print(random_check_dates.shape)
#std = StandardScaler()

random_check_std = std.transform(random_check_set)


#%%
#X_check, y_check = fetch_batch(test_set, 1, l, w, pred_window)
img_dir = "../../../images/random_new/only_attack/"
with tf.Session() as sess:
    saver.restore(sess,directory)
    
    for i in range(5):
        X_check, y_check, date_check = fetch_batch(random_check_std,random_check_dates, 1, l, w, pred_window)
        #X_check, y_check = fetch_batch(test_set, 1, l, w, pred_window)
        prediction = sess.run(output,feed_dict={X:X_check[:,:,:,2:], y: y_check})
        plt.figure()
        plt.plot_date(date_check.reshape(-1),y_check[0,:],xdate=True,label='actual',ls="-")
        plt.plot_date(date_check.reshape(-1),prediction[0,:],xdate=True,label='predictions',ls="-")
        plt.xticks(rotation="vertical")
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Attack')
        save_fig(i,img_dir)