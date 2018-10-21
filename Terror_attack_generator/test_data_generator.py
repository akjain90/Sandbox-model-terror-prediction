# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 23:36:20 2018

@author: akjai
"""

import numpy as np
from model_variables import *
import matplotlib.pyplot as plt

date = pd.date_range(start=start_date,end=end_date)
date_len = len(date)
print(date_len)
t = np.linspace(0,10,date_len)
x = np.sin(2*np.pi*50*t)+np.sin(2*np.pi*100*t)+t
attack_df = pd.DataFrame({'Attacks':x,'Date':date})
attack_df.to_csv('../ML_predictor/test_sin_data.csv')