# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:30:06 2018

@author: akjai
"""
# Imports
#%%
import numpy as np
import datetime as dt
import pandas as pd
#%%
start_date = dt.date(1990,1,1)  # dt.date(year, month, day)
end_date = dt.date(2015,12,31)
num_holidays = 12
first_full_moon = dt.date(1990,1,2)
#randomness = 5

full_moon_booster = 0.1
holiday_booster = 0.1
# resource pool: rp
rp_init = 4500
rp_deposit_per_day = 500
rp_withdraw_per_casuality = 500
#%%
# lone wolf attack variables
# lone wolf: lw
lw_casualities_low = 1
lw_casualities_high = 5
lw_attack_dist = [0.01,0.99]
lw_attack_possi = [True,False]
lw_casualities = [1,2,3,4]
lw_casualities_dist = [0.5,0.25,0.15,0.1]
