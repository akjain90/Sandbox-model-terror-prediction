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
import pandas as pd

def main():
   #num_days = 10000
   #attack = np.zeros(num_days)
   ga = generate_data()
   date = pd.date_range(start=start_date,end=end_date)
   
   fp_temp = []
   holiday_data = []
   full_moons = ga.all_fm
   holidays = ga.holidays
   full_moon_array = np.zeros(len(date))
   holiday_array = np.zeros(len(date))
   temp_1 = start_date
   #print(int(start_date in all_full_moon))
   index = 0
   #print(all_full_moon)
   while temp_1<=end_date:
       full_moon_array[index] = int(temp_1 in full_moons)
       holiday_array[index] = int(temp_1 in holidays)
       ga.loneWolf_attack()
       ga.set_holi_attack_fact(temp_1)
       ga.set_fm_attack_fact(temp_1)
       ga.set_rp_attack_fact()
       ga.set_tg_casualities()
       fp_temp.append(ga.tg_casualities+ga.total_lw_attack)
       temp_1+=dt.timedelta(1)
       index += 1
   #print((full_moon_array))
   #print(len(pd.date_range(start=start_date,end=end_date)))
   attack_df = pd.DataFrame({'Date':date,
                             'Num_attacks':fp_temp,
                             'Full_moons' : full_moon_array,
                             'Holidays': holiday_array})
#   print(attack_df.head(10))
#   attack_df.to_csv('../ML_predictor/data.csv')
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