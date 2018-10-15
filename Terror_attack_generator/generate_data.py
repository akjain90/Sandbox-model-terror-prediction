from model_variables import *
import numpy as np
import datetime as dt

# tg: terror group
# lw: lone wolf
# rp: resource pool
class generate_data:
    
    def __init__(self):
        self.total_lw_attack = 0  # total attacks of all lone wolf
        self.rp = rp_init  # resource pool's current value
        self.tg_casualities = 0  # casualities by terrorist group attack
        self.rp_attack_fact = 0  # terror grpup attack probablity 
                                 # due to resource pool value
        self.fm_attack_fact = 0  # probablity contribution of full moon 
                                 # to terror group attack
        self.holi_attack_fact = 0  # probablity contribution of holiday 
                                   # to terror group attack
        self.all_fm = self.set_fm()  # list of all full moons
        self.holidays = self.set_holidays(num_holidays)  # list of all holidays
        
    def set_holidays(self,N):
        day_range = np.linspace(1,30,30)
        month_range = np.linspace(1,12,12)
        days = np.random.choice(day_range,N).astype(np.int32)
        months = np.random.choice(month_range,N).astype(np.int32)
        year = start_date.year
        holidays = []
        while year <= end_date.year:
            for day,month in zip(days,months):
                try:
                    holidays.append(dt.date(year,month,day))
                except:
                    #print('Exception is generated')
                    None
            year+=1
        return(np.array(holidays))
        
    def set_fm(self):
        full_moons = []
        fm_date = first_fm
        while fm_date<end_date:
            full_moons.append(fm_date)
            fm_date = fm_date + dt.timedelta(days=30)
        return np.array(full_moons)
    
    def loneWolf_attack(self):
        self.total_lw_attack = 0
        #np.random.seed(56)
        num_lw = np.random.randint(low = 0, high = 5)
        num_casualities_per_lw = np.zeros(num_lw)
        for lw in range(num_lw):
            is_lw_attacked = np.random.choice(a = attack_possi,
                                                p = lw_attack_dist)
            if is_lw_attacked:
                num_casualities_per_lw[lw] = np.random.choice(a = lw_casualities,
                                                       p = lw_casualities_dist)
        self.total_lw_attack = np.sum(num_casualities_per_lw)
        #return self.total_lw_attack
        
    def set_rp_attack_fact(self):
        self.rp += rp_deposit_per_day
        self.rp_attack_fact = (self.rp-5000)*3/100000
        
    def set_fm_attack_fact(self, today):
        dist_to_fm = np.absolute(today-self.all_fm)
        closest_dist_to_fm = np.min(dist_to_fm).days
        if closest_dist_to_fm <= 5:
            self.fm_attack_fact = fm_attack_dist[closest_dist_to_fm]
        else:
            self.fm_attack_fact = 0
        
        
    def set_holi_attack_fact(self, today):
        dist_to_holi = np.absolute(today-self.holidays)
        closest_dist_to_holi = np.min(dist_to_holi).days
        if closest_dist_to_holi <= 5:
            self.holi_attack_fact = holi_attack_dist[closest_dist_to_holi]
        else:
            self.holi_attack_fact = 0
        
        
    def rp_manager(self):
        self.rp = self.rp-self.tg_casualities * rp_withdraw_per_casuality
        
    
    def set_tg_casualities(self):
        #global count
        self.tg_casualities = 0
        total_attack_fact = self.fm_attack_fact+self.holi_attack_fact+self.rp_attack_fact
        if total_attack_fact<0:
            total_attack_fact = 0
        elif total_attack_fact>1:
            total_attack_fact = 1
        is_tg_attacked = np.random.choice(a = attack_possi,
                                          p = [total_attack_fact,1-total_attack_fact])
        if is_tg_attacked:
            self.tg_casualities = np.random.randint(20,50)
            #self.tg_casualities = 1
        self.rp_manager()
