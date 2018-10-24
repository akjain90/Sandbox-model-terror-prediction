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

#%%
data = pd.read_csv("../Philippinen.csv",index_col=0)

data_val = data.values
print(len(data_val))
#%%
train_len = np.floor_divide(90*len(data_val),100)
train_set = data_val[:train_len,0:1]
train_date = data_val[:train_len,1]
test_set = data_val[train_len:,0:1]
test_date = data_val[train_len:,1]

std = StandardScaler()

train_std = std.fit_transform(train_set)
test_std = std.transform(test_set)
print(len(train_std))
print(len(test_std))
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
directory = '../saved_model/Philippinen/only_attack/'


X = tf.placeholder(dtype = tf.float32, 
                   shape = (None, l,w,c), name="X")
y = tf.placeholder(dtype = tf.float32,
                   shape = (None,pred_window), name="y")
training = tf.placeholder_with_default(False,[],name="training")

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

optimizer = tf.train.AdamOptimizer(learning_rate=0.0008)

training_op = optimizer.minimize(mse)

saver = tf.train.Saver()
init = tf.global_variables_initializer()

#%%
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        X_batch, y_batch, _ = fetch_batch(train_std,train_date, batch_size, l, w, pred_window)
        #X_batch, y_batch,_ = fetch_batch(train_set,train_date, batch_size, l, w, pred_window)
        sess.run(training_op, feed_dict = {X:X_batch, y: y_batch})
        if epoch%50==0:
            train_error = sess.run(mse, feed_dict = {X:X_batch, y: y_batch})
            test_x, test_y,_ = fetch_batch(test_std,test_date, 1, l, w, pred_window)
            #test_x, test_y,_ = fetch_batch(test_set,test_date, 1, l, w, pred_window)
            test_error = sess.run(mse, feed_dict = {X:test_x, y: test_y})
            print("Epoch: ",epoch, " Training error: ", train_error, " Test error: ", test_error)
    saver.save(sess,directory)
    #plt.figure()
    #plt.plot()
    

#print(mse)

#%%
#X_check, y_check = fetch_batch(test_set, 1, l, w, pred_window)
img_dir = "../../../images/Philippinen/only_attack/"
with tf.Session() as sess:
    saver.restore(sess,directory)
    
    for i in range(5):
        X_check, y_check,date_check = fetch_batch(test_std,test_date, 1, l, w, pred_window)
        #X_check, y_check,date_check = fetch_batch(test_set,test_date, 1, l, w, pred_window)
        prediction = sess.run(output,feed_dict={X:X_check, y: y_check})
        plt.figure()
        plt.plot_date(date_check.reshape(-1),y_check[0,:],xdate=True,label='actual',ls='-')
        plt.plot_date(date_check.reshape(-1),prediction[0,:],xdate=True,label='predictions',ls='-')
        plt.xticks(rotation="vertical")
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Attack')
        save_fig(i,img_dir)