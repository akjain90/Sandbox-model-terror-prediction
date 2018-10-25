# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:58:08 2018

@author: jain
"""
import tensorflow as tf

def predict(data,date,sess,directory,prediction_window = 30):
    
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(directory + '.meta')
#        sess.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
#        saver = tf.train.import_meta_graph(directory + '.meta')
        saver.restore(sess,directory)
        output = graph.get_tensor_by_name('output/BiasAdd:0')
        X = data[:-prediction_window].reshape(1,10,10,1)
        prediction = sess.run(output,feed_dict={'X:0':X})
    return prediction,data[-prediction_window:],date[-prediction_window:]