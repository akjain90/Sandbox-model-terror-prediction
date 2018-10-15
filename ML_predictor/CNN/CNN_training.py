import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
#%%

def fetch_batch(data, batch_size, l, w, pred_window):
    num_steps = l*w
    data_len, features = data.shape
    end = data_len-num_steps-pred_window
    index = np.random.randint(0,end,batch_size)
    X = []
    y = []
    for i in index:
        temp_X = data[i:i+num_steps,:].reshape(l,w,features)
        temp_y = data[i+num_steps:i+num_steps+pred_window,0].reshape(-1)
        X.append(temp_X)
        y.append(temp_y)
    return np.array(X), np.array(y)
#%%
data = pd.read_csv("../only_full_moon.csv",index_col=0)

data_val = data.values
print(len(data_val))
#%%
train_len = np.floor_divide(70*len(data_val),100)
train_set = data_val[:train_len,1:]
test_set = data_val[train_len:,1:]

std = StandardScaler()

train_std = std.fit_transform(train_set)
test_std = std.transform(test_set)
#%%
# graph definition
tf.reset_default_graph()

l = 10
w = 10
c = 3
pred_window = 30
num_epoch = 2000
batch_size = 200


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
        #X_batch, y_batch = fetch_batch(train_std, batch_size, l, w, pred_window)
        X_batch, y_batch = fetch_batch(train_set, batch_size, l, w, pred_window)
        sess.run(training_op, feed_dict = {X:X_batch, y: y_batch})
        if epoch%50==0:
            train_error = sess.run(mse, feed_dict = {X:X_batch, y: y_batch})
            #test_x, test_y = fetch_batch(test_std, 1, l, w, pred_window)
            test_x, test_y = fetch_batch(test_set, 1, l, w, pred_window)
            test_error = sess.run(mse, feed_dict = {X:test_x, y: test_y})
            print("Epoch: ",epoch, " Training error: ", train_error, " Test error: ", test_error)
    saver.save(sess,'../saved_model/model')
    #plt.figure()
    #plt.plot()
    

#print(mse)

#%%
#X_check, y_check = fetch_batch(test_set, 1, l, w, pred_window)
with tf.Session() as sess:
    saver.restore(sess,'../saved_model/model')
    #X_check, y_check CNN= fetch_batch(test_std, 1, l, w, pred_window)
    X_check, y_check = fetch_batch(test_set, 1, l, w, pred_window)
    prediction = sess.run(output,feed_dict={X:X_check, y: y_check})
    plt.figure()
    plt.plot(1+y_check[0,:])
    plt.plot(prediction[0,:])