# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:48:33 2018

@author: jain
"""
import numpy as np

def normed_loss(s1,s2):
    return np.linalg.norm(s1-s2)