# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:33:50 2018

@author: jain
"""
from modules.fetch_batch import fetch_batch
import tensorflow as tf

def training(data,model_dir,num_epoch=20,batch_size=200,n_steps=100,n_inputs=1,n_neurons=50,n_outputs=1,n_layers=3):
    
    tf.reset_default_graph()

    learning_rate = 0.0001

    X = tf.placeholder(tf.float32, shape=(None,n_steps,n_inputs),name='X')
    y = tf.placeholder(tf.float32, shape=(None,n_steps,n_outputs), name = 'y')
    
    cell = [tf.contrib.rnn.LSTMCell(num_units=n_neurons) for layer in range(n_layers)]
    multicell = tf.contrib.rnn.MultiRNNCell(cell)
    rnn_outputs, states = tf.nn.dynamic_rnn(multicell, inputs=X, dtype=tf.float32)
    stacked_rnn_outputs = tf.reshape(tensor=rnn_outputs, shape=(-1,n_neurons))
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, units=n_outputs,name='stacked_output')
    outputs = tf.reshape(stacked_outputs, shape=(-1,n_steps,n_outputs),name='output')
    mse = tf.reduce_mean(tf.square(outputs-y),name='mse')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    training_loss = []
    epoch_count = []
    with tf.Session() as sess:
        init.run()
        for epoch in range(num_epoch):
            X_train, y_train = fetch_batch(data.reshape(-1,1), batch_size,n_steps)
            sess.run(training_op,feed_dict={X:X_train, y:y_train})
            if epoch%10==0:
                loss = sess.run(mse,feed_dict={X:X_train, y:y_train})
                training_loss.append(loss)
                epoch_count.append(epoch)
                if(epoch%100==0):
                    print('Epoch: ',epoch,' Training loss: ',loss)
        saver.save(sess,model_dir) 
    

    return sess, training_loss, epoch_count
        