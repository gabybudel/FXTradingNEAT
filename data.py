# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:51:09 2018

@author: gabys
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats 



fpath = ''


def log_returns(prices):
    return np.log(prices/prices.shift(1))      
            

def write_stats_table(df, path):
    to_write = ["\\begin{table}[]", 
                "\\centering", 
                "\\caption{My Title}", 
                "\\label{tab:mylabel}",  
                "\\begin{tabular}{l" + ''.join("r" for d in df.columns) + "}", 
                "\\toprule",
                ''.join((" & " + d) for d in df.columns) + " \\\\",
                "\\midrule",
                "Mean" + ''.join((" & " + "%.2f" % np.mean(df.iloc[:,j])) for j in range(0,df.shape[1]))  + " \\\\",
                "St.Dev." + ''.join((" & " + "%.2f" % np.std(df.iloc[:,j], ddof = 1)) for j in range(0,df.shape[1]))  + " \\\\",
                "Skewness" + ''.join((" & " + "%.2f" % scipy.stats.skew(df.iloc[:,j].dropna())) for j in range(0,df.shape[1]))  + " \\\\",
                "Kurtosis" + ''.join((" & " + "%.2f" % scipy.stats.kurtosis(df.iloc[:,j].dropna())) for j in range(0,df.shape[1]))  + " \\\\",
                "$Q_{0.05}$" + ''.join((" & " + "%.2f" % np.percentile(df.iloc[:,j].dropna(), 5)) for j in range(0,df.shape[1]))  + " \\\\",
                "Median" + ''.join((" & " + "%.2f" % np.percentile(df.iloc[:,j].dropna(), 50)) for j in range(0,df.shape[1]))  + " \\\\",
                "$Q_{0.95}$" + ''.join((" & " + "%.2f" % np.percentile(df.iloc[:,j].dropna(), 95)) for j in range(0,df.shape[1]))  + " \\\\",                
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                ]
    file = open(path, "w")
    for line in to_write:
        file.write(line + '\n')
    file.close()

    
def write_returns_table(df, path):
    to_write = ["\\begin{table}[]", 
                "\\centering", 
                "\\caption{My Title}", 
                "\\label{tab:mylabel}",  
                "\\begin{tabular}{l" + ''.join("r" for d in df.columns) + "}", 
                "\\toprule",
                ''.join((" & " + d) for d in df.columns) + " \\\\",
                "\\midrule",
                "Mean" + ''.join((" & " + "%.2f" % (12*np.mean(df.iloc[:,j]))) for j in range(0,df.shape[1]))  + " \\\\",
                "St.Dev." + ''.join((" & " + "%.2f" % (np.sqrt(12)*np.std(df.iloc[:,j], ddof = 1))) for j in range(0,df.shape[1]))  + " \\\\",
                "Skewness" + ''.join((" & " + "%.2f" % scipy.stats.skew(df.iloc[:,j].dropna())) for j in range(0,df.shape[1]))  + " \\\\",
                "Kurtosis" + ''.join((" & " + "%.2f" % scipy.stats.kurtosis(df.iloc[:,j].dropna())) for j in range(0,df.shape[1]))  + " \\\\",
                "Sharpe Ratio" + ''.join((" & " + "%.2f" % ((12*np.mean(df.iloc[:,j]))/(np.sqrt(12)*np.std(df.iloc[:,j], ddof = 1)))) for j in range(0,df.shape[1]))  + " \\\\",
                "$Q_{0.05}$" + ''.join((" & " + "%.2f" % np.percentile(12*df.iloc[:,j].dropna(), 5)) for j in range(0,df.shape[1]))  + " \\\\",
                "Median" + ''.join((" & " + "%.2f" % np.percentile(12*df.iloc[:,j].dropna(), 50)) for j in range(0,df.shape[1]))  + " \\\\",
                "$Q_{0.95}$" + ''.join((" & " + "%.2f" % np.percentile(12*df.iloc[:,j].dropna(), 95)) for j in range(0,df.shape[1]))  + " \\\\",                
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                ]
    file = open(path, "w")
    for line in to_write:
        file.write(line + '\n')
    file.close()


crncy = {
'UKDOLLR(ER)' : 'GBP',
'SWISSF$(ER)' : 'CHF',
'KORSWO$(ER)' : 'KRW',
'EUDOLLR(ER)' : 'EUR',
'MEXPES$(ER)' : 'MXN',
'COMRAN$(ER)' : 'ZAR',
'AUSTDOI(ER)' : 'AUD',
'NORKRO$(ER)' : 'NOK',
'SWEKRO$(ER)' : 'SEK',
'SINGDO$(ER)' : 'SGD',
'TAIWDO$(ER)' : 'TWD',
'DANISH$(ER)' : 'DKK',
'NZDOLLI(ER)' : 'NZD',
'BRACRU$(ER)' : 'BRL',
'CNDOLL$(ER)' : 'CAD',
'JAPAYE$(ER)' : 'JPY',
'UKDOLLR(EB)' : 'GBP',
'SWISSF$(EB)' : 'CHF',
'KORSWO$(EB)' : 'KRW',
'EUDOLLR(EB)' : 'EUR',
'MEXPES$(EB)' : 'MXN',
'COMRAN$(EB)' : 'ZAR',
'AUSTDOI(EB)' : 'AUD',
'NORKRO$(EB)' : 'NOK',
'SWEKRO$(EB)' : 'SEK',
'SINGDO$(EB)' : 'SGD',
'TAIWDO$(EB)' : 'TWD',
'DANISH$(EB)' : 'DKK',
'NZDOLLI(EB)' : 'NZD',
'BRACRU$(EB)' : 'BRL',
'CNDOLL$(EB)' : 'CAD',
'JAPAYE$(EB)' : 'JPY',
'UKDOLLR(EO)' : 'GBP',
'SWISSF$(EO)' : 'CHF',
'KORSWO$(EO)' : 'KRW',
'EUDOLLR(EO)' : 'EUR',
'MEXPES$(EO)' : 'MXN',
'COMRAN$(EO)' : 'ZAR',
'AUSTDOI(EO)' : 'AUD',
'NORKRO$(EO)' : 'NOK',
'SWEKRO$(EO)' : 'SEK',
'SINGDO$(EO)' : 'SGD',
'TAIWDO$(EO)' : 'TWD',
'DANISH$(EO)' : 'DKK',
'NZDOLLI(EO)' : 'NZD',
'BRACRU$(EO)' : 'BRL',
'CNDOLL$(EO)' : 'CAD',
'JAPAYE$(EO)' : 'JPY',
}

fwdrt = {
'EUDOL1F(ER)' : 'EUR',
'USJPY1F(ER)' : 'JPY',
'USKRW1F(ER)' : 'KRW',
'UKUSD1F(ER)' : 'GBP',
'USCAD1F(ER)' : 'CAD',
'USDKK1F(ER)' : 'DKK',
'USMXN1F(ER)' : 'MXN',
'USZAR1F(ER)' : 'ZAR',
'USSEK1F(ER)' : 'SEK',
'USTWD1F(ER)' : 'TWD',
'USCHF1F(ER)' : 'CHF',
'USBRL1F(ER)' : 'BRL',
'USSGD1F(ER)' : 'SGD',
'USNOK1F(ER)' : 'NOK',
'BBAUD1F(ER)' : 'AUD',
'BBNZD1F(ER)' : 'NZD',
'EUDOL1F(EB)' : 'EUR',
'USJPY1F(EB)' : 'JPY',
'USKRW1F(EB)' : 'KRW',
'UKUSD1F(EB)' : 'GBP',
'USCAD1F(EB)' : 'CAD',
'USDKK1F(EB)' : 'DKK',
'USMXN1F(EB)' : 'MXN',
'USZAR1F(EB)' : 'ZAR',
'USSEK1F(EB)' : 'SEK',
'USTWD1F(EB)' : 'TWD',
'USCHF1F(EB)' : 'CHF',
'USBRL1F(EB)' : 'BRL',
'USSGD1F(EB)' : 'SGD',
'USNOK1F(EB)' : 'NOK',
'BBAUD1F(EB)' : 'AUD',
'BBNZD1F(EB)' : 'NZD',
'EUDOL1F(EO)' : 'EUR',
'USJPY1F(EO)' : 'JPY',
'USKRW1F(EO)' : 'KRW',
'UKUSD1F(EO)' : 'GBP',
'USCAD1F(EO)' : 'CAD',
'USDKK1F(EO)' : 'DKK',
'USMXN1F(EO)' : 'MXN',
'USZAR1F(EO)' : 'ZAR',
'USSEK1F(EO)' : 'SEK',
'USTWD1F(EO)' : 'TWD',
'USCHF1F(EO)' : 'CHF',
'USBRL1F(EO)' : 'BRL',
'USSGD1F(EO)' : 'SGD',
'USNOK1F(EO)' : 'NOK',
'BBAUD1F(EO)' : 'AUD',
'BBNZD1F(EO)' : 'NZD',
}

bb_vol = {
        'USDCADV1Y' : 'CAD',
        'USDCHFV1Y' : 'CHF',
        'USDDKKV1Y' : 'DKK',
        'USDSEKV1Y' : 'SEK',
        'USDSGDV1Y' : 'SGD',
        'USDKRWV1Y' : 'KRW',
        'USDTWDV1Y' : 'TWD',
        'USDZARV1Y' : 'ZAR',
        'USDBRLV1Y' : 'BRL',
        'USDMXNV1Y' : 'MXN',
        'USDJPYV1Y' : 'JPY',
        'USDNOKV1Y' : 'NOK',
        'EURUSDV1Y' : 'EUR',
        'AUDUSDV1Y' : 'AUD',
        'GBPUSDV1Y' : 'GBP',
        'NZDUSDV1Y' : 'NZD',
}

trading_days = 252
fx = pd.ExcelFile('Processed/FX_data.xlsx')

# Exchange rates
fx_er = fx.parse('SP_ER_monthly', index_col = 0, parse_dates = True)
fx_er = fx_er.sort_index()
fx_er.index = pd.to_datetime(fx_er.index)
fx_er.columns = [crncy.get(c) for c in fx_er.columns.values]  # change variable names
fx_er = 1/fx_er  # IMPORTANT!
fx_eb = fx.parse('SP_EB_monthly', index_col = 0, parse_dates = True)
fx_eb = fx_eb.sort_index()
fx_eb.index = pd.to_datetime(fx_eb.index)
fx_eb.columns = [crncy.get(c) for c in fx_eb.columns.values]  # change variable names
temp = 1/fx_eb  # IMPORTANT!
fx_eo = fx.parse('SP_EO_monthly', index_col = 0, parse_dates = True)
fx_eo = fx_eo.sort_index()
fx_eo.index = pd.to_datetime(fx_eo.index)
fx_eo.columns = [crncy.get(c) for c in fx_eo.columns.values]  # change variable names
fx_eb = 1/fx_eo  # IMPORTANT!
fx_eo = temp


# Forward rates
fwd_er = fx.parse('1F_ER_monthly', index_col = 0, parse_dates = True)
fwd_er = fwd_er.sort_index() 
fwd_er.index = pd.to_datetime(fwd_er.index)
fwd_er.columns = [fwdrt.get(f) for f in fwd_er.columns.values]  # change variable names
fwd_er = 1/fwd_er  # IMPORTANT!
fwd_eb = fx.parse('1F_EB_monthly', index_col = 0, parse_dates = True)
fwd_eb = fwd_eb.sort_index() 
fwd_eb.index = pd.to_datetime(fwd_eb.index)
fwd_eb.columns = [fwdrt.get(f) for f in fwd_eb.columns.values]  # change variable names
temp = 1/fwd_eb  # IMPORTANT!
fwd_eo = fx.parse('1F_EO_monthly', index_col = 0, parse_dates = True)
fwd_eo = fwd_eo.sort_index() 
fwd_eo.index = pd.to_datetime(fwd_eo.index)
fwd_eo.columns = [fwdrt.get(f) for f in fwd_eo.columns.values]  # change variable names
fwd_eb = 1/fwd_eo  # IMPORTANT!
fwd_eo = temp
yfwd_er = fx.parse('1F_ER_monthly', index_col = 0, parse_dates = True)
yfwd_er = yfwd_er.sort_index() 
yfwd_er.index = pd.to_datetime(yfwd_er.index)
yfwd_er.columns = [fwdrt.get(f) for f in yfwd_er.columns.values]  # change variable names
yfwd_er = 1/yfwd_er  # IMPORTANT!

# Daily exchange rates
fx_daily = fx.parse('SP_ER_daily', index_col = 0, parse_dates = True)
fx_daily = fx_daily.sort_index()
fx_daily.index = pd.to_datetime(fx_daily.index)
fx_daily.columns = [crncy.get(c) for c in fx_daily.columns.values]  # change variable names
fx_daily = 1/fx_daily  # IMPORTANT!
fwd_daily = fx.parse('FW_ER_daily', index_col = 0, parse_dates = True)
fwd_daily = fwd_daily.sort_index()
fwd_daily.index = pd.to_datetime(fwd_daily.index)
fwd_daily.columns = [crncy.get(c) for c in fwd_daily.columns.values]  # change variable names
fwd_daily = 1/fwd_daily  # IMPORTANT!

# Implied volatilities Bloomberg
imp_vol = fx.parse('ImpVol_Bloom', index_col = 0, parse_dates = True)
imp_vol = imp_vol.sort_index() 
imp_vol.index = pd.to_datetime(imp_vol.index)
imp_vol.columns = [bb_vol.get(c) for c in imp_vol.columns.values]  # change variable names

# Recession Periods
nber = fx.parse('Recession', index_col = 0, parse_dates = True)
nber = nber.sort_index()
nber.index = pd.to_datetime(nber.index)


first_date = fx_er.index[145]  # corresponds to 1997-01-23 (inclusive)
last_date = fx_er.index[-1]  # corresponds to 2018-04-23 (not inclusive)

# Subset to complete data
fx_er = fx_er.loc[(fx_er.index >= first_date) & (fx_er.index < last_date),:]
fx_eb = fx_eb.loc[(fx_eb.index >= first_date) & (fx_eb.index < last_date),:]
fx_eo = fx_eo.loc[(fx_eo.index >= first_date) & (fx_eo.index < last_date),:]
fwd_er = fwd_er.loc[(fwd_er.index >= first_date) & (fwd_er.index < last_date),:]
fwd_eb = fwd_eb.loc[(fwd_eb.index >= first_date) & (fwd_eb.index < last_date),:]
fwd_eo = fwd_eo.loc[(fwd_eo.index >= first_date) & (fwd_eo.index < last_date),:]
yfwd_er = yfwd_er.loc[(yfwd_er.index >= first_date) & (yfwd_er.index < last_date),:]
imp_vol = imp_vol.loc[imp_vol.index >= (first_date - np.timedelta64(15, 'D')),:]
nber = nber.loc[(nber.index >= first_date) & (nber.index < last_date)]

# Calculate returns
returns_monthly = log_returns(fx_er)
returns_daily = log_returns(fx_daily)
ret_ex_long = (fx_eb - fwd_eo.shift(1))/fx_er.shift(1)
ret_ex_short = (fx_eo - fwd_eb.shift(1))/fx_er.shift(1)
ret_ex = (fx_er - fwd_er.shift(1))/fx_er.shift(1)
ret_exchange_long = (fx_eb - fx_eo.shift(1))/fx_er.shift(1)
ret_exchange_short = (fx_eo - fx_eb.shift(1))/fx_er.shift(1)
ret_exchange = (fx_er - fx_er.shift(1))/fx_er.shift(1)

# Replace weird values
returns_daily[returns_daily == 0] = float('NaN')  # replace returns of 0 by NaN
returns_daily = returns_daily.dropna(axis = 0, how = 'all')  # remove rows with only NaNs

# Expected realized volatility under physical measure
returns_daily_sq = returns_daily.pow(2)
temp = [pd.DataFrame(np.sqrt(returns_daily_sq.loc[(returns_daily_sq.index > (d - np.timedelta64(1, 'Y'))) & \
                                     (returns_daily_sq.index <= d)].sum(axis=0))).transpose() \
    for d in fx_er.index]
exp_vol_ph = pd.concat(temp)*100  # times 100 for percentages
exp_vol_ph.index = fx_er.index
exp_vol_ph[exp_vol_ph == 0] = float('NaN')
exp_vol_ph = exp_vol_ph[exp_vol_ph.index >= (np.min(imp_vol.index) - np.timedelta64(15, 'D'))]
exp_vol_ph = exp_vol_ph[exp_vol_ph.index < np.max(imp_vol.index)]

# Force different index
imp_vol.index = exp_vol_ph.index

# Volatility risk premium
vrp = exp_vol_ph - imp_vol

###### Strategies #######

fw_premium = (fwd_er - fx_er)/fx_er
ret_three_months = (fx_er - fx_er.shift(3))/fx_er.shift(3)
car_signals = pd.DataFrame(0, index = fx_er.index, columns = fx_er.columns)
mom_signals = pd.DataFrame(0, index = fx_er.index, columns = fx_er.columns)
vrp_signals = pd.DataFrame(0, index = fx_er.index, columns = fx_er.columns)
bench_ret = pd.DataFrame(index = fx_er.index, columns = ['CAR', 'MOM', 'VRP'])
exch_ret = pd.DataFrame(index = fx_er.index, columns = ['CAR', 'MOM', 'VRP'])
pf_size = 0.2
for row in fx_er.index:
    n_car = np.int64(np.round(len(fwd_er.loc[row,:].dropna())*pf_size))
    n_mom = np.min([np.int64(np.round(len(ret_three_months.loc[row,:].dropna())*pf_size)),
                    np.int64(np.round(len(fwd_er.loc[row,:].dropna())*pf_size))])
    n_vrp = np.min([np.int64(np.round(len(vrp.loc[row,:].dropna())*pf_size)), 
                    np.int64(np.round(len(fwd_er.loc[row,:].dropna())*pf_size))])
    
    # Carry trade
    sorted_premia = np.sort(fw_premium.loc[row,:].dropna())
    lb = sorted_premia[n_car-1]
    ub = sorted_premia[-n_car]
    car_signals.loc[row,fw_premium.loc[row,:] <= lb] = 1  # long in lowest forward premia/highest differential
    car_signals.loc[row,fw_premium.loc[row,:] >= ub] = -1
    
    # Momentum
    if n_mom > 0:
        # Only calculate return for months where we have three preceding months
        sorted_ret = np.sort(ret_three_months.loc[row,~np.isnan(fwd_er.loc[row,:])].dropna())
        lb = sorted_ret[n_mom-1]
        ub = sorted_ret[-n_mom]
        mom_signals.loc[row,(ret_three_months.loc[row,~np.isnan(fwd_er.loc[row,:])] >= ub) & \
                        (~np.isnan(fwd_er.loc[row,:]))] = 1  # long in highest momentum, only when forward data is available
        mom_signals.loc[row,(ret_three_months.loc[row,~np.isnan(fwd_er.loc[row,:])] <= lb) & \
                        (~np.isnan(fwd_er.loc[row,:]))] = -1
    
    # VRP
    if n_vrp > 0:
        # Only calculate return for months where we have three preceding months
        sorted_ret = np.sort(vrp.loc[row,~np.isnan(fwd_er.loc[row,:])].dropna())
        lb = sorted_ret[n_vrp-1]
        ub = sorted_ret[-n_vrp]
        vrp_signals.loc[row,(vrp.loc[row,~np.isnan(fwd_er.loc[row,:])] >= ub) & \
                        (~np.isnan(fwd_er.loc[row,:]))] = 1  # long in highest momentum, only when forward data is available
        vrp_signals.loc[row,(vrp.loc[row,~np.isnan(fwd_er.loc[row,:])] <= lb) & \
                        (~np.isnan(fwd_er.loc[row,:]))] = -1

bench_ret['CAR'] = [np.mean(ret_ex_long.loc[d,car_signals.shift(1).loc[d,:] == 1]) - \
         np.mean(ret_ex_short.loc[d,car_signals.shift(1).loc[d,:] == -1]) for d in ret_ex_long.index]
bench_ret['MOM'] = [np.mean(ret_ex_long.loc[d,mom_signals.shift(1).loc[d,:] == 1]) - \
         np.mean(ret_ex_short.loc[d,mom_signals.shift(1).loc[d,:] == -1]) for d in ret_ex_long.index]
bench_ret['VRP'] = [np.mean(ret_ex_long.loc[d,vrp_signals.shift(1).loc[d,:] == 1]) - \
         np.mean(ret_ex_short.loc[d,vrp_signals.shift(1).loc[d,:] == -1]) for d in ret_ex_long.index]

exch_ret['CAR'] = [np.mean(ret_exchange_long.loc[d,car_signals.shift(1).loc[d,:] == 1]) - 
         np.mean(ret_exchange_short.loc[d,car_signals.shift(1).loc[d,:] == -1]) for d in ret_exchange_long.index]
exch_ret['MOM'] = [np.mean(ret_exchange_long.loc[d,mom_signals.shift(1).loc[d,:] == 1]) - 
         np.mean(ret_exchange_short.loc[d,mom_signals.shift(1).loc[d,:] == -1]) for d in ret_exchange_long.index]
exch_ret['VRP'] = [np.mean(ret_exchange_long.loc[d,vrp_signals.shift(1).loc[d,:] == 1]) - 
         np.mean(ret_exchange_short.loc[d,vrp_signals.shift(1).loc[d,:] == -1]) for d in ret_exchange_long.index]

#write_returns_table(100*bench_ret.loc[:,['CAR','MOM','VRP']],"Output/strategy_returns.txt")

# Cumulative
strat_cum = bench_ret.cumsum()


def minmax_scale(df, split):
    return (df - np.min(df[df.index < split]))/(np.max(df[df.index < split]) \
            - np.min(df[df.index < split]))

# Create train and test set
split_date = fx_er.index[-60]  # last 5 years of data
lagged1_ret = (fx_er - fx_er.shift(1))/fx_er.shift(1)
lagged2_ret = (fx_er - fx_er.shift(2))/fx_er.shift(2)
lagged3_ret = (fx_er - fx_er.shift(3))/fx_er.shift(3)
lagged1_ret_ex = (fx_er - fwd_er.shift(1))/fx_er.shift(1)
lagged2_ret_ex = (fx_er - fwd_er.shift(2))/fx_er.shift(2)
lagged3_ret_ex = (fx_er - fwd_er.shift(3))/fx_er.shift(3)
curr_vol = exp_vol_ph
lagged1_vol = exp_vol_ph.shift(1)
lagged2_vol = exp_vol_ph.shift(2)
lagged3_vol = exp_vol_ph.shift(3)
curr_vrp = vrp
curr_imp_vol = imp_vol
curr_fw_prem = fw_premium
curr_spread = (fx_eo - fx_eb)/fx_eo


# Scale
lagged1_ret_sc = minmax_scale(lagged1_ret, split_date)
lagged2_ret_sc = minmax_scale(lagged2_ret, split_date)
lagged3_ret_sc = minmax_scale(lagged3_ret, split_date)
lagged1_ret_ex_sc = minmax_scale(lagged1_ret_ex, split_date)
lagged2_ret_ex_sc = minmax_scale(lagged2_ret_ex, split_date)
lagged3_ret_ex_sc = minmax_scale(lagged3_ret_ex, split_date)
curr_vol_sc = minmax_scale(curr_vol, split_date)
lagged1_vol_sc = minmax_scale(lagged1_vol, split_date)
lagged2_vol_sc = minmax_scale(lagged2_vol, split_date)
lagged3_vol_sc = minmax_scale(lagged3_vol, split_date)
curr_imp_vol_sc = minmax_scale(imp_vol, split_date)
curr_vrp_sc = minmax_scale(curr_vrp, split_date)
curr_fw_prem_sc = minmax_scale(curr_fw_prem, split_date)
curr_spread_sc = minmax_scale(curr_spread, split_date)
nber_sc = nber  # already 0s and 1s

