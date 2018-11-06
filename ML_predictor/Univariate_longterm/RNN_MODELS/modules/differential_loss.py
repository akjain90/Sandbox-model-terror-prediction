# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:57:05 2018

@author: jain
"""

import numpy as np

def differential_loss(s1,s2):
    return np.abs(np.sum(s1)-np.sum(s2))