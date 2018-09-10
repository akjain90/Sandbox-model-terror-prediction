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
       ga.loneWolf_attack()
       ga.set_holi_attack_fact(temp_1)
       ga.set_fm_attack_fact(temp_1)
       ga.set_rp_attack_fact()
       ga.set_tg_casualities()
       fp_temp.append(ga.tg_casualities+ga.total_lw_attack)
       temp_1+=dt.timedelta(1)
   plt.plot(fp_temp)
   
## Script to check holidays attack factor
## even full moon attack factor can be set and both the attack factor can be added and plotted
#   fp_temp = []
#   temp_1 = start_date
#   while temp_1<end_date:
#       ga.set_holi_attack_fact(temp_1)
#       fp_temp.append(ga.holi_attack_fact)
#       temp_1+=dt.timedelta(1)
#   plt.plot(fp_temp)
   
## Script to check full moon attack factor
#   fp_temp = []
#   temp_1 = start_date
#   while temp_1<end_date:
#       ga.set_fm_attack_fact(temp_1)
#       fp_temp.append(ga.fm_attack_fact)
#       temp_1+=dt.timedelta(1)
#   plt.plot(fp_temp)
   
## script to check lone wolf attack per day
#   for day in range(num_days):
#       attack[day] = ga.loneWolf_attack()
#   print (attack)
#   print(np.sum(attack))
#   plt.plot(attack)
if __name__ == "__main__":
    main()