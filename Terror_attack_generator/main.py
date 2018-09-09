# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 02:04:56 2018

@author: akjai
"""
#%%
from generate_data import *
import matplotlib.pyplot as plt


def main():
   num_days = 1000
   attack = np.zeros(num_days)
   ga = generate_data()
   for day in range(num_days):
       attack[day] = ga.loneWolf_attack()
   print (attack)
   print(np.sum(attack))
   plt.plot(attack)
if __name__ == "__main__":
    main()