# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:58:08 2018

@author: jain
"""
import tensorflow as tf
import numpy as np

def predict(data,date,sess,directory,n_steps,n_inputs,prediction_window = 30):
    
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(directory + '.meta')
        graph = tf.get_default_graph()
        saver.restore(sess,directory)
        output = graph.get_tensor_by_name('output:0')
        predict_accum = data[:n_steps].tolist()
        for i in range (prediction_window):
            #new series generation feeding the predicted value
            X = np.array(predict_accum[-n_steps:]).reshape(1,n_steps,n_inputs)
            #feeding the true value
            #X = np.array(data[i:i+n_steps]).reshape(1,n_steps,n_inputs)
            prediction = sess.run(output,feed_dict={'X:0':X})
            predict_accum.append(prediction[:,-1,:])
    return np.array(predict_accum[-prediction_window:]).reshape(-1,1),data[-prediction_window:],date[-prediction_window:]