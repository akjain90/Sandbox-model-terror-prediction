# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 02:04:56 2018

@author: akjai
"""
#%%
from generate_data import *
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np


def main():
   num_days = 10000
   attack = np.zeros(num_days)
   ga = generate_data()
   fp_temp = []
   temp_1 = start_date
   while temp_1<end_date:
       ga.set_fm_attack_fact(temp_1)
       fp_temp.append(ga.fm_attack_fact)
       temp_1+=dt.timedelta(1)
   plt.plot(fp_temp)
#print(temp_2)
#   for day in range(num_days):
#       attack[day] = ga.loneWolf_attack()
#   print (attack)
#   print(np.sum(attack))
#   plt.plot(attack)
if __name__ == "__main__":
    main()