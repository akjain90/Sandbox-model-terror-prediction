# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:33:50 2018

@author: jain
"""
from modules.save_fig import save_fig
from modules.fetch_batch import fetch_batch
import tensorflow as tf

def training(data,date,model_dir,num_epoch=20,batch_size=200,pred_window = 30):
    
    tf.reset_default_graph()
    
    l = 10
    w = 10
    c = 1
     
    X = tf.placeholder(dtype = tf.float32, 
                       shape = (None, l,w,c), name="X")
    y = tf.placeholder(dtype = tf.float32,
                       shape = (None,pred_window), name="y")
    
    conv_1 = tf.layers.conv2d(X,
                              filters=8,
                              kernel_size=(5,5),
                              strides=1,
                              padding="same",
                              activation=tf.nn.relu, 
                              name = "conv_1")
    #print(conv_1)
    conv_2 = tf.layers.conv2d(conv_1,
                              filters=16,
                              kernel_size=(5,5),
                              strides=1,
                              padding="same",
                              activation=tf.nn.relu, 
                              name = "conv_2")
    #print(conv_2)
#    pool_2 = tf.layers.max_pooling2d(conv_2, 
#                                     pool_size=(3,3),
#                                     strides=2,
#                                     name = "max_pool_2")
    
    flatten = tf.layers.flatten(conv_2)
    #print(flatten)
    dense_1 = tf.layers.dense(flatten, units=800, activation=tf.nn.relu, name="dense_1")
    
    dense_2 = tf.layers.dense(dense_1, units=200, activation=tf.nn.relu, name="dense_2")
    
    dense_3 = tf.layers.dense(dense_2, units=80, activation=tf.nn.relu, name="dense_3")
    
    output = tf.layers.dense(dense_3, units=pred_window, name="output")
    
    mse = tf.reduce_mean(tf.square(output-y),name="MSE")
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0008)
    
    training_op = optimizer.minimize(mse)
    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
    training_loss = []
    epoch_count = []

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epoch):
            X_batch, y_batch,_ = fetch_batch(data,date, batch_size, l, w, pred_window)
            sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
            if epoch%10==0:
                loss = sess.run(mse, feed_dict = {X:X_batch, y:y_batch})
                training_loss.append(loss)
                epoch_count.append(epoch)
                if(epoch%200==0):
                    print('Epoch:', epoch,' Training loss:',loss)
        saver.save(sess,model_dir)
    return sess, training_loss, epoch_count
        