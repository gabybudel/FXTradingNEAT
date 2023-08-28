# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:23:03 2018

@author: gabys
"""

import numpy as np
import pandas as pd
import os
import neat
import sys
import pickle
import scipy
from scipy import stats
import visualize


pf_size = 0.2

filehandler = open('winner_net.pckl', 'rb')
best_net = pickle.load(filehandler)
config_path = 'config-feedforward'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

temp = lagged1_ret_sc + lagged2_ret_sc + lagged3_ret_sc + \
lagged1_ret_ex_sc + lagged2_ret_ex_sc + lagged3_ret_ex_sc + curr_vol_sc + \
lagged1_vol_sc + lagged2_vol_sc + lagged3_vol_sc + curr_imp_vol_sc + \
curr_vrp_sc + curr_fw_prem_sc + curr_spread_sc
complete_test = (temp == temp)

test_dates = fx_er.index[fx_er.index >= split_date]
ml_signals = pd.DataFrame(0, index = fx_er.index, columns = fx_er.columns)
for d in ml_signals.index:
    complete_crncy = complete_test.columns[complete_test.loc[d,:]]
    if len(complete_crncy) >= 3:
        activations = pd.DataFrame(index = complete_crncy,
                                   columns = ['Activation'])
        for c in activations.index:
            xi = (lagged1_ret_sc.loc[d,c],lagged2_ret_sc.loc[d,c],\
                  lagged3_ret_sc.loc[d,c],lagged1_ret_ex_sc.loc[d,c],\
                  lagged2_ret_ex_sc.loc[d,c],lagged3_ret_ex_sc.loc[d,c],\
                  curr_vol_sc.loc[d,c],lagged1_vol_sc.loc[d,c],\
                  lagged2_vol_sc.loc[d,c],lagged3_vol_sc.loc[d,c],\
                  curr_imp_vol_sc.loc[d,c],curr_vrp_sc.loc[d,c],\
                  curr_fw_prem_sc.loc[d,c],curr_spread_sc.loc[d,c],nber_sc.loc[d,'RECESS'])
            output = best_net.activate(xi)
            activations.loc[c,'Activation'] = output[0] 
            sorted_act = np.sort(activations['Activation'])  # index 0 is smallest
            n_invest = np.int64(np.round(len(activations.index)*pf_size))
            lb = sorted_act[n_invest-1]
            ub = sorted_act[-n_invest]
            long = activations.index[activations['Activation'] >= ub]
            short = activations.index[activations['Activation'] <= lb]
            ml_signals.loc[d,[(c in long) for c in ml_signals.columns]] = 1
            ml_signals.loc[d,[(c in short) for c in ml_signals.columns]] = -1
    
# Calculate return
ml_returns = [(np.mean(ret_ex_long.loc[d,ml_signals.shift(1).loc[d,:] == 1]) - \
 np.mean(ret_ex_short.loc[d,ml_signals.shift(1).loc[d,:] == -1])) for d in ml_signals.index]

strat_ret = pd.DataFrame(index = fx_er.index)
strat_ret['CAR'] = bench_ret['CAR']
strat_ret['MOM'] = bench_ret['MOM']
strat_ret['VRP'] = bench_ret['VRP']
strat_ret['ML'] = ml_returns

test_ret = pd.DataFrame(index = fx_er.index[fx_er.index >= split_date])
test_ret['CAR'] = bench_ret.loc[bench_ret.index >= split_date,'CAR']
test_ret['MOM'] = bench_ret.loc[bench_ret.index >= split_date,'MOM']
test_ret['VRP'] = bench_ret.loc[bench_ret.index >= split_date,'VRP']
test_ret['ML'] = strat_ret['ML'][strat_ret.index >= split_date]

# Calculate exchange rate returns
ml_exch_returns = [(np.mean(ret_exchange_long.loc[d,ml_signals.shift(1).loc[d,:] == 1]) - \
 np.mean(ret_exchange_short.loc[d,ml_signals.shift(1).loc[d,:] == -1])) for d in ml_signals.index]

strat_ret_exchange = pd.DataFrame(index = fx_er.index)
strat_ret_exchange['CAR'] = exch_ret['CAR']
strat_ret_exchange['MOM'] = exch_ret['MOM']
strat_ret_exchange['VRP'] = exch_ret['VRP']
strat_ret_exchange['ML'] = ml_exch_returns

test_ret_exchange = pd.DataFrame(index = fx_er.index[fx_er.index >= split_date])
test_ret_exchange['CAR'] = exch_ret.loc[exch_ret.index >= split_date,'CAR']
test_ret_exchange['MOM'] = exch_ret.loc[exch_ret.index >= split_date,'MOM']
test_ret_exchange['VRP'] = exch_ret.loc[exch_ret.index >= split_date,'VRP']
test_ret_exchange['ML'] = strat_ret_exchange['ML'][strat_ret.index >= split_date]

writer = pd.ExcelWriter('Strategy_Returns.xlsx')
strat_ret.to_excel(writer,'Strat_Ret')
test_ret.to_excel(writer,'Test_Ret')
nber.to_excel(writer,'NBER')
writer.save()