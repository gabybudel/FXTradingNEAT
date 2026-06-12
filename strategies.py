# -*- coding: utf-8 -*-
"""Return and volatility computations (interactive snippet).

Subset of the calculations in ``data.py``, kept as a working note. NOT a
standalone script: it references ``fx_er``, ``fx_daily``, ``fwd_er``,
``imp_vol``, and ``trading_days`` from a session in which ``data.py`` was
run. Note the realized-volatility window here uses ``trading_days`` calendar
days, whereas ``data.py`` uses a full calendar year.
"""

import numpy as np
import pandas as pd


def log_returns(prices):
    """Return one-period log returns of a price DataFrame/Series."""
    return np.log(prices/prices.shift(1))

# Calculate returns
returns_monthly = log_returns(fx_er)
returns_daily = log_returns(fx_daily)
ret_ex = (fx_er - fwd_er)/fx_er.shift(1)

# Weird values
returns_daily[returns_daily == 0] = float('NaN')  # replace returns of 0 by NaN
returns_daily = returns_daily.dropna(axis = 0, how = 'all')  # remove rows with only NaNs

# Expected realized volatility under physical measure
returns_daily_sq = returns_daily.pow(2)
temp = [pd.DataFrame(np.sqrt(returns_daily_sq.loc[(returns_daily_sq.index > (d - np.timedelta64(trading_days, 'D'))) & \
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