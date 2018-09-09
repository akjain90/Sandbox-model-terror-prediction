from model_variables import *
import numpy as np
import datetime as dt
class generate_data:
    def __init__(self):
        self.total_lw_attack = 0
        self.rp = rp_init
        # terror group: tg
        self.tg_casualities = 0
        self.rp_attack_fact = 0
        self.full_moon_attack_fact = 0
        self.holiday_attack_fact = 0
        self.all_full_moon = self.set_full_moon()
        
    def set_full_moon(self):
        full_moons = []
        full_moon_date = first_full_moon
        while full_moon_date<end_date:
            full_moons.append(full_moon_date)
            full_moon_date = full_moon_date + dt.timedelta(days=30)
        return np.array(full_moons)
    
    def loneWolf_attack(self):
        self.total_lw_attack = 0
        #np.random.seed(56)
        # loneWolf : lw
        num_lw = np.random.randint(low = lw_casualities_low, high = lw_casualities_high)
        num_casualities_per_lw = np.zeros(num_lw)
        for lw in range(num_lw):
            is_lw_attacked = np.random.choice(a = lw_attack_possi,
                                                p = lw_attack_dist)
            if is_lw_attacked:
                num_casualities_per_lw[lw] = np.random.choice(a = lw_casualities,
                                                       p = lw_casualities_dist)
        self.total_lw_attack = np.sum(num_casualities_per_lw)
        #return self.total_lw_attack
        
    def deposit_in_rp(self):
        self.rp = rp_deposit_per_day
        self.rp_attack_fact = (self.rp-5000)*3/100000
        
    def set_full_moon_attack_fact(today):
        None       
        
        
    def rp_manager(self):
        self.rp = self.rp-self.tg_casualities * rp_withdraw_per_casuality
        
    
    def set_tg_attacks(self):
        None
    
    def fill_attacks(self):
        None
