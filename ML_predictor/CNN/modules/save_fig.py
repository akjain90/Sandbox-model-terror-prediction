# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:49:33 2018

@author: akjai
"""
import os
import matplotlib.pyplot as plt
""" Save figure """
def save_fig(fig_id,directory, tight_layout=True):
    path = os.path.join(directory+ str(fig_id) + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)