# -*- coding: utf-8 -*-
"""Data preparation for the NEAT FX trading project.

Loads spot, forward, implied-volatility, and NBER recession data from
``Processed/FX_data.xlsx``, then computes (as module-level variables):

* mid/bid/offer spot and 1-month forward rates, expressed as USD per unit of
  foreign currency (raw quotes are foreign currency per USD and are inverted
  on load; inversion swaps the roles of bid and offer);
* monthly and daily log returns, and excess returns versus lagged forwards
  (with and without bid/offer transaction costs);
* 1-year rolling realized volatility, implied volatility, and the volatility
  risk premium (``vrp = exp_vol_ph - imp_vol``);
* benchmark long-short strategy returns (``bench_ret`` / ``exch_ret`` with
  columns CAR, MOM, VRP), each going long/short the top/bottom ``pf_size``
  fraction of currencies, signals lagged one month;
* min-max-scaled NEAT input features (``*_sc``), scaled using only data
  before ``split_date`` (the final 60 months are the test set).

This script is intended to be run once in an interactive session
(``%run data.py``); the evaluation scripts (``evaluate_ml.py``,
``evaluate_secondrun.py``, ``plots.py``, ``strategies.py``) consume the
variables it leaves in the namespace.
"""

import numpy as np
import pandas as pd
import scipy
from scipy import stats


def log_returns(prices):
    """Return one-period log returns of a price DataFrame/Series."""
    return np.log(prices/prices.shift(1))


def write_stats_table(df, path):
    """Write a LaTeX (booktabs) table of summary statistics of ``df``.

    Reports mean, standard deviation, skewness, kurtosis, and the 5%/50%/95%
    quantiles per column. Statistics are computed on non-missing values.
    """
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
    with open(path, "w") as f_out:
        for line in to_write:
            f_out.write(line + '\n')


def write_returns_table(df, path):
    """Write a LaTeX (booktabs) table of annualized return statistics.

    ``df`` must contain monthly returns: means and quantiles are annualized
    by a factor 12, standard deviations by sqrt(12), and the Sharpe ratio is
    the ratio of the two (zero risk-free benchmark is implicit because the
    inputs are excess returns).
    """
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
    with open(path, "w") as f_out:
        for line in to_write:
            f_out.write(line + '\n')


# Mapping of Datastream spot-rate mnemonics to ISO currency codes.
# Suffixes: (ER) = mid rate, (EB) = bid, (EO) = offer.
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

# Mapping of Datastream 1-month forward-rate mnemonics to ISO currency codes.
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

# Mapping of Bloomberg 1-year implied-volatility tickers to ISO currency codes.
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


def load_sheet(xlfile, sheet, name_map=None):
    """Parse ``sheet``, sort by date, and map raw tickers to ISO codes."""
    df = xlfile.parse(sheet, index_col=0, parse_dates=True)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    if name_map is not None:
        df.columns = [name_map.get(c) for c in df.columns.values]
    return df


fx = pd.ExcelFile('Processed/FX_data.xlsx')

# Spot exchange rates. Raw quotes are foreign currency per USD; invert to USD
# per foreign currency. Inversion swaps bid and offer: the USD-per-FC bid
# price is the reciprocal of the FC-per-USD offer price, and vice versa.
fx_er = 1/load_sheet(fx, 'SP_ER_monthly', crncy)
fx_eb = 1/load_sheet(fx, 'SP_EO_monthly', crncy)
fx_eo = 1/load_sheet(fx, 'SP_EB_monthly', crncy)

# 1-month forward rates, inverted (and bid/offer swapped) the same way
fwd_er = 1/load_sheet(fx, '1F_ER_monthly', fwdrt)
fwd_eb = 1/load_sheet(fx, '1F_EO_monthly', fwdrt)
fwd_eo = 1/load_sheet(fx, '1F_EB_monthly', fwdrt)
yfwd_er = fwd_er.copy()

# Daily exchange rates
fx_daily = 1/load_sheet(fx, 'SP_ER_daily', crncy)
fwd_daily = 1/load_sheet(fx, 'FW_ER_daily', crncy)

# Implied volatilities Bloomberg (already in volatility units, no inversion)
imp_vol = load_sheet(fx, 'ImpVol_Bloom', bb_vol)

# NBER recession indicator (0/1, column 'RECESS')
nber = load_sheet(fx, 'Recession')


# Sample window with (near-)complete cross-sectional coverage
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

# Align implied-vol observation dates to the realized-vol (month-end) index.
# NOTE: this is a positional overwrite, not a merge — it assumes both series
# cover the same months in the same order.
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
                        (~np.isnan(fwd_er.loc[row,:]))] = 1  # long in highest VRP, only when forward data is available
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
    """Min-max scale ``df`` using only observations strictly before ``split``.

    Using the training-period extrema for the full sample avoids look-ahead
    bias in the test period (test values may fall outside [0, 1]).
    """
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

