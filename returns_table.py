# -*- coding: utf-8 -*-
"""LaTeX table writer for two panels of monthly returns.

Standalone copy of ``write_returns_table`` from ``evaluate_ml.py``: reports
annualized mean, standard deviation, Sharpe ratio, and quantiles plus
skewness, kurtosis, and first-order autocorrelation, side by side for two
return panels.
"""

import numpy as np
import scipy
from scipy import stats


def write_returns_table(df1, df2, path):
    """Write a two-panel LaTeX (booktabs) table of annualized return stats."""
    to_write = ["\\begin{table}[]",
                "\\centering",
                "\\caption{My Title}",
                "\\label{tab:mylabel}",
                "\\begin{tabular}{l" + ''.join("r" for d in df1.columns) + "r" + ''.join("r" for d in df2.columns) + "}",
                "\\toprule",
                ''.join((" & " + d) for d in df1.columns) + " & " + ''.join((" & " + d) for d in df2.columns) + " \\\\",
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
    with open(path, "w") as f_out:
        f_out.writelines(line + '\n' for line in to_write)