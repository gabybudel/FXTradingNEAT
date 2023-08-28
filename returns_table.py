# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 08:56:15 2018

@author: gabys
"""

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