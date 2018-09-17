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
data = pd.read_csv("../data.csv",index_col=0)

data_val = data.values
#%%
train_len = np.floor_divide(70*len(attacks),100)
train_set = data_val[:train_len,1:]
test_set = data_val[train_len:,1:]

#std = StandardScaler()

train_std = std.fit_transform(train_set)
test_std = std.transform(test_set)
#%%
# graph definition
tf.reset_default_graph()

l = 20
w = 20
c = 3
pred_window = 30
num_epoch = 4000
batch_size = 20


X = tf.placeholder(dtype = tf.float32, 
                   shape = (None, l,w,c), name="X")
y = tf.placeholder(dtype = tf.float32,
                   shape = (None,pred_window), name="y")

conv_1 = tf.layers.conv2d(X,
                          filters=16,
                          kernel_size=(3,3),
                          strides=1,
                          padding="same",
                          activation=tf.nn.relu, 
                          name = "conv_1")

pool_1 = tf.layers.max_pooling2d(conv_1, 
                                 pool_size=(3,3),
                                 strides=2,
                                 name = "max_pool_1")

conv_2 = tf.layers.conv2d(pool_1,
                          filters=32,
                          kernel_size=(3,3),
                          strides=1,
                          padding="same",
                          activation=tf.nn.relu, 
                          name = "conv_2")

pool_2 = tf.layers.max_pooling2d(conv_2, 
                                 pool_size=(3,3),
                                 strides=2,
                                 name = "max_pool_2")

conv_3 = tf.layers.conv2d(pool_2,
                          filters=64,
                          kernel_size=(2,2),
                          strides=1,
                          padding="same",
                          activation=tf.nn.relu, 
                          name = "conv_3")

pool_3 = tf.layers.max_pooling2d(conv_3, 
                                 pool_size=(2,2),
                                 strides=2,
                                 name = "max_pool_3")

flatten = tf.layers.flatten(pool_3)

dense_1 = tf.layers.dense(flatten, units=100, activation=tf.nn.relu, name="dense_1")


output = tf.layers.dense(dense_1, units=pred_window, name="output")

mse = tf.reduce_mean(tf.square(output-y),name="MSE")

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

#%%
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        X_batch, y_batch = fetch_batch(train_std, batch_size, l, w, pred_window)
        sess.run(training_op, feed_dict = {X:X_batch, y: y_batch})
        if epoch%100==0:
            train_error = sess.run(mse, feed_dict = {X:X_batch, y: y_batch})
            test_x, test_y = fetch_batch(test_std, 1, l, w, pred_window)
            test_error = sess.run(mse, feed_dict = {X:test_x, y: test_y})
            print("Epoch: ",epoch, " Training error: ", train_error, " Test error: ", test_error)
    

#print(mse)

#%%
#X, y = (fetch_batch(train_set, 10, 20, 20, pred_window))