# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:23:02 2018

@author: TerriZ
"""

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
#from scipy.stats import t
from scipy import stats
import seaborn as sns

import matplotlib.dates as mdates
import datetime as dt

file_location = "Output/Strategy_Returns.xlsx" 

f = pd.ExcelFile(file_location)
strat_ret = f.parse('Strat_Ret', index_col = 0, parse_dates = True)
test_ret = f.parse('Test_Ret', index_col = 0, parse_dates = True)
nber = f.parse('NBER', index_col = 0, parse_dates = True)
split_date = strat_ret.index[-60]  # last 5 years of data

plt.axvspan(*mdates.datestr2num(['3/1/2001', '11/1/2001']), color=sns.xkcd_rgb['grey'], alpha=0.5, lw=0)
plt.axvspan(*mdates.datestr2num(['12/1/2007', '6/1/2009']), color=sns.xkcd_rgb['grey'], alpha=0.5, lw=0)
plt.plot(np.cumprod(1+strat_ret))
plt.xlim([strat_ret.index[0], mdates.datestr2num(['04/23/2018'])])
plt.axvspan(*mdates.datestr2num(['4/23/2013', '4/23/2013']), color=sns.xkcd_rgb['black'], alpha=1, lw=1)
plt.legend(strat_ret.columns)
plt.show()

plt.plot(np.cumprod(1+test_ret))
plt.xlim([mdates.datestr2num(['04/23/2013']), mdates.datestr2num(['04/23/2018'])])
plt.legend(test_ret.columns)
plt.show()


avg_imp = imp_vol.mean(axis = 1)
avg_vrp = vrp.mean(axis = 1)
fig, ax1 = plt.subplots()
ax1.axvspan(*mdates.datestr2num(['3/1/2001', '11/1/2001']), color=sns.xkcd_rgb['grey'], alpha=0.5, lw=0)
ax1.axvspan(*mdates.datestr2num(['12/1/2007', '6/1/2009']), color=sns.xkcd_rgb['grey'], alpha=0.5, lw=0)
vrp_plot = ax1.plot(avg_vrp, 'darkblue', label = 'VRP')
#ax1.xlim([strat_ret.index[0], mdates.datestr2num(['04/23/2018'])])
ax2 = ax1.twinx()
iv_plot = ax2.plot(avg_imp, 'darkorange', label = 'IV')
plt.xlim([strat_ret.index[0], mdates.datestr2num(['04/23/2018'])])
lns = vrp_plot + iv_plot
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
fig.tight_layout()
plt.show()