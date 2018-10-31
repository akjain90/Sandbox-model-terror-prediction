# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:58:08 2018

@author: jain
"""
import tensorflow as tf

def predict(data,date,sess,directory,prediction_window = 30):
    