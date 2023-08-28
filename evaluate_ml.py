# -*- coding: utf-8 -*-
"""
Created on Sun May 13 11:05:04 2018

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

fpath = 'C:/Users/gabys/Documents/Erasmus/MScFI Thesis/Data/Output/Second Run'
os.chdir(fpath)

pf_size = 0.2

filehandler = open('winner_net.pckl', 'rb')
best_net = pickle.load(filehandler)
config_path = ''
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
temp = pd.DataFrame(index = fx_er.index)
temp['ML'] = ml_returns

strat_ret = pd.DataFrame(index = fx_er.index[fx_er.index < split_date])
strat_ret['CAR'] = bench_ret.loc[bench_ret.index < split_date,'CAR']
strat_ret['MOM'] = bench_ret.loc[bench_ret.index < split_date,'MOM']
strat_ret['VRP'] = bench_ret.loc[bench_ret.index < split_date,'VRP']
strat_ret['ML'] = temp.loc[temp.index < split_date,'ML']

test_ret = pd.DataFrame(index = fx_er.index[fx_er.index >= split_date])
test_ret['CAR'] = bench_ret.loc[bench_ret.index >= split_date,'CAR']
test_ret['MOM'] = bench_ret.loc[bench_ret.index >= split_date,'MOM']
test_ret['VRP'] = bench_ret.loc[bench_ret.index >= split_date,'VRP']
test_ret['ML'] = temp['ML'][temp.index >= split_date]

# Calculate exchange rate returns
ml_exch_returns = [(np.mean(ret_exchange_long.loc[d,ml_signals.shift(1).loc[d,:] == 1]) - \
 np.mean(ret_exchange_short.loc[d,ml_signals.shift(1).loc[d,:] == -1])) for d in ml_signals.index]
temp_exch = pd.DataFrame(index = fx_er.index)
temp_exch['ML'] = ml_exch_returns

strat_ret_exchange = pd.DataFrame(index = fx_er.index)
strat_ret_exchange['CAR'] = exch_ret.loc[exch_ret.index < split_date,'CAR']
strat_ret_exchange['MOM'] = exch_ret.loc[exch_ret.index < split_date,'MOM']
strat_ret_exchange['VRP'] = exch_ret.loc[exch_ret.index < split_date,'VRP']
strat_ret_exchange['ML'] = temp_exch.loc[temp_exch.index < split_date,'ML']

test_ret_exchange = pd.DataFrame(index = fx_er.index[fx_er.index >= split_date])
test_ret_exchange['CAR'] = exch_ret.loc[exch_ret.index >= split_date,'CAR']
test_ret_exchange['MOM'] = exch_ret.loc[exch_ret.index >= split_date,'MOM']
test_ret_exchange['VRP'] = exch_ret.loc[exch_ret.index >= split_date,'VRP']
test_ret_exchange['ML'] = temp_exch['ML'][temp_exch.index >= split_date]

writer = pd.ExcelWriter('Strategy_Returns.xlsx')
strat_ret.to_excel(writer,'Strat_Ret')
test_ret.to_excel(writer,'Test_Ret')
nber.to_excel(writer,'NBER')
writer.save()

f = pd.ExcelFile('FILEPATH')
strat_ret = f.parse('Strat_Ret', index_col = 0, parse_dates = True)
test_ret = f.parse('Test_Ret', index_col = 0, parse_dates = True)
nber = f.parse('NBER', index_col = 0, parse_dates = True)
split_date = strat_ret.index[-60]  # last 5 years of data

def write_returns_table(df1, df2, path):
    to_write = ["\\begin{table}[]", 
                "\\centering", 
                "\\caption{My Title}", 
                "\\label{tab:mylabel}",  
                "\\begin{tabular}{l" + ''.join("r" for d in df1.columns) + "r" + ''.join("r" for d in df2.columns) + "}", 
                "\\toprule",
                ''.join((" & " + d) for d in df1.columns) + " & " + ''.join((" & " + d) for d in df1.columns) + " \\\\",
                "\\midrule",
                "Mean" + ''.join((" & " + "%.2f" % (12*np.mean(df1.iloc[:,j]))) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % (12*np.mean(df2.iloc[:,j]))) for j in range(0,df2.shape[1]))  + " \\\\",
                "St.Dev." + ''.join((" & " + "%.2f" % (np.sqrt(12)*np.std(df1.iloc[:,j], ddof = 1))) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % (np.sqrt(12)*np.std(df2.iloc[:,j], ddof = 1))) for j in range(0,df2.shape[1])) + " \\\\",
                "Skewness" + ''.join((" & " + "%.2f" % scipy.stats.skew(df1.iloc[:,j].dropna())) for j in range(0,df1.shape[1]))  + " & " + ''.join((" & " + "%.2f" % scipy.stats.skew(df2.iloc[:,j].dropna())) for j in range(0,df2.shape[1])) + " \\\\",
                "Kurtosis" + ''.join((" & " + "%.2f" % scipy.stats.kurtosis(df1.iloc[:,j].dropna())) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % scipy.stats.kurtosis(df2.iloc[:,j].dropna())) for j in range(0,df2.shape[1])) + " \\\\",
                "$SR$" + ''.join((" & " + "%.2f" % ((12*np.mean(df1.iloc[:,j]))/(np.sqrt(12)*np.std(df1.iloc[:,j], ddof = 1)))) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % ((12*np.mean(df2.iloc[:,j]))/(np.sqrt(12)*np.std(df2.iloc[:,j], ddof = 1)))) for j in range(0,df2.shape[1])) + " \\\\",
                "$AC$" + ''.join((" & " + "%.2f" % df1.iloc[:,j].dropna().autocorr()) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % df2.iloc[:,j].dropna().autocorr()) for j in range(0,df2.shape[1])) + " \\\\",
                "$Q_{0.05}$" + ''.join((" & " + "%.2f" % np.percentile(12*df1.iloc[:,j].dropna(), 5)) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % np.percentile(12*df2.iloc[:,j].dropna(), 5)) for j in range(0,df2.shape[1])) + " \\\\",
                "Median" + ''.join((" & " + "%.2f" % np.percentile(12*df1.iloc[:,j].dropna(), 50)) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % np.percentile(12*df2.iloc[:,j].dropna(), 50)) for j in range(0,df2.shape[1])) + " \\\\",
                "$Q_{0.95}$" + ''.join((" & " + "%.2f" % np.percentile(12*df1.iloc[:,j].dropna(), 95)) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % np.percentile(12*df2.iloc[:,j].dropna(), 95)) for j in range(0,df2.shape[1])) + " \\\\",                
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                ]
    file = open(path, "w")
    file.writelines(line + '\n' for line in to_write)
    file.close()
    
def write_stats_table(df1, df2, path):
    to_write = ["\\begin{table}[]", 
                "\\centering", 
                "\\caption{My Title}", 
                "\\label{tab:mylabel}",  
                "\\begin{tabular}{l" + ''.join("r" for d in df1.columns) + "r" + ''.join("r" for d in df2.columns) + "}", 
                "\\toprule",
                ''.join((" & " + d) for d in df1.columns) + " & " + ''.join((" & " + d) for d in df1.columns) + " \\\\",
                "\\midrule",
                "Mean" + ''.join((" & " + "%.2f" % (np.mean(df1.iloc[:,j]))) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % (np.mean(df2.iloc[:,j]))) for j in range(0,df2.shape[1]))  + " \\\\",
                "St.Dev." + ''.join((" & " + "%.2f" % (np.std(df1.iloc[:,j], ddof = 1))) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % (np.std(df2.iloc[:,j], ddof = 1))) for j in range(0,df2.shape[1])) + " \\\\",
                "Skewness" + ''.join((" & " + "%.2f" % scipy.stats.skew(df1.iloc[:,j].dropna())) for j in range(0,df1.shape[1]))  + " & " + ''.join((" & " + "%.2f" % scipy.stats.skew(df2.iloc[:,j].dropna())) for j in range(0,df2.shape[1])) + " \\\\",
                "Kurtosis" + ''.join((" & " + "%.2f" % scipy.stats.kurtosis(df1.iloc[:,j].dropna())) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % scipy.stats.kurtosis(df2.iloc[:,j].dropna())) for j in range(0,df2.shape[1])) + " \\\\",
                "$AC$" + ''.join((" & " + "%.2f" % df1.iloc[:,j].dropna().autocorr()) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % df2.iloc[:,j].dropna().autocorr()) for j in range(0,df2.shape[1])) + " \\\\",
                "$Q_{0.05}$" + ''.join((" & " + "%.2f" % np.percentile(df1.iloc[:,j].dropna(), 5)) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % np.percentile(df2.iloc[:,j].dropna(), 5)) for j in range(0,df2.shape[1])) + " \\\\",
                "Median" + ''.join((" & " + "%.2f" % np.percentile(df1.iloc[:,j].dropna(), 50)) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % np.percentile(df2.iloc[:,j].dropna(), 50)) for j in range(0,df2.shape[1])) + " \\\\",
                "$Q_{0.95}$" + ''.join((" & " + "%.2f" % np.percentile(df1.iloc[:,j].dropna(), 95)) for j in range(0,df1.shape[1])) + " & " + ''.join((" & " + "%.2f" % np.percentile(df2.iloc[:,j].dropna(), 95)) for j in range(0,df2.shape[1])) + " \\\\",                
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                ]
    file = open(path, "w")
    file.writelines(line + '\n' for line in to_write)
    file.close()

write_returns_table(100*strat_ret, 100*strat_ret_exchange, "ret_strategy_returns_table.txt")
write_returns_table(100*test_ret, 100*test_ret_exchange, "ret_test_returns_table.txt")

table_data = pd.DataFrame({'VRP': vrp.stack(), 'IV': imp_vol.stack(), 'RV': exp_vol_ph.stack()}) 
table_test_data = pd.DataFrame({'VRP': vrp[vrp.index >= split_date].stack(), 'IV': imp_vol[vrp.index >= split_date].stack(), 'RV': exp_vol_ph[vrp.index >= split_date].stack()}) 
write_stats_table(table_data,table_test_data,"Output/vol_table.txt")
