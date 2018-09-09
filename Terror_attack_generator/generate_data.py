from model_variables import *
import numpy as np
class generate_data:
    def __init__(self):
        None
    
    def loneWolf_attack(self):
        loneWolf_attacks = [1,2,3,4]
        loneWolf_attack_dist = [0.4,0.3,0.2,0.1]
        num_loneWolf = np.random.randint(low = 1, high = 5)
        lone_wolf_attack_dist = [0.01,0.99]
        attack_possi = [True,False]
        num_attacks_loneWolf = np.zeros(num_loneWolf)
        for lone_wolf in range(num_loneWolf):
            is_loneWolf_attacked = np.random.choice(a = attack_possi,
                                                p = lone_wolf_attack_dist)
            if is_loneWolf_attacked:
                num_attacks_loneWolf[lone_wolf] = np.random.choice(a = loneWolf_attacks,
                                                       p = loneWolf_attack_dist)
        total_loneWolf_attack = np.sum(num_attacks_loneWolf)
        return total_loneWolf_attack
    
    def fill_attacks(self):
        None
