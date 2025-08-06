#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 11:06:07 2025

@author: katharinafriedel
"""

# --- IMPORT LIBRARIES ---

# import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
from scipy.stats import levene
import tabulate

# --- DATA DOWNLOAD & PREPARATIONS ---

# download historical FX data (only adjusted close prices)
pairs = ['EURUSD=X', 'GBPUSD=X', 'JPY=X', 'AUDUSD=X', 'USDCHF=X', 'USDCAD=X']
data = yf.download(pairs, start='2006-01-01', end='2020-12-31', auto_adjust=True)['Close'].dropna()

# calculate daily log returns to ensure time-additive properties
returns = np.log(data/data.shift(1))

# define financial crisis periods manually based on known global market shocks
crisis_mask = (
    ((returns.index>='2008-09-01') & (returns.index<='2009-06-30')) |
    ((returns.index>='2020-03-01') & (returns.index<='2020-05-31'))
)

# split data into calm and crisis periods
returns_crisis = returns.loc[crisis_mask]
returns_calm = returns.loc[~crisis_mask]

# --- CORRELATION ANALYSIS ---

# calculate correlation matrices for period
corr_crisis = returns_crisis.corr()
corr_calm = returns_calm.corr()

# plot correlation heatmaps
plt.figure(figsize=(8,6))
sns.heatmap(corr_crisis, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('FX Correlations During Crisis')
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(corr_calm, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('FX Correlations During Calm')
plt.show()

# compare correlations between calm and crisis using the Fisher Z-test
def fisher_z(r):
    return 0.5 * np.log((1+r)/(1-r))

def compare_correlations(r1, n1, r2, n2):
    z1, z2 = fisher_z(r1), fisher_z(r2)
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z1 - z2)/se
    p = 2*(1-norm.cdf(abs(z)))
    return z, p

# matrix to store results
corr_diff_results = []

for c1 in returns.columns:
    for c2 in returns.columns:
        if c1 >= c2:
            continue
        r_calm = corr_calm.loc[c1, c2]
        r_crisis = corr_crisis.loc[c1, c2]
        n_calm = len(returns_calm)
        n_crisis = len(returns_crisis)
        z_val, p_val = compare_correlations(r_calm, n_calm, r_crisis, n_crisis)
        corr_diff_results.append({
            'pair': f"{c1} vs {c2}",
            'corr_calm': r_calm,
            'corr_crisis': r_crisis,
            'z_stat': z_val,
            'p_value': p_val
        })

df_corr_diff = pd.DataFrame(corr_diff_results)
print(df_corr_diff.sort_values('p_value'))

# build final results table with only statistically significant differences (p < 0.05)
df_corr_diff = df_corr_diff.dropna().reset_index(drop=True)
df_corr_diff = df_corr_diff[df_corr_diff['p_value']<0.05]
df_corr_diff = df_corr_diff.sort_values('p_value')

print(tabulate.tabulate(df_corr_diff, headers='keys', tablefmt='github', floatfmt=".3f"))

# download as csv file
df_corr_diff.to_csv('significant_correlation_changes.csv', index=False)

# rolling correlation between key pairs (EUR/USD & GBP/USD)
rolling_corr = returns['EURUSD=X'].rolling(60).corr(returns['GBPUSD=X'])
rolling_corr.plot(figsize=(10,5), title='60-Day Rolling Correlation: EUR/USD vs GBP/USD')
plt.axvspan('2008-09-01', '2009-06-30', color='red', alpha=0.2, label='2008 Crisis')
plt.axvspan('2020-03-01', '2020-05-31', color='orange', alpha=0.2, label='COVID Crisis')
plt.legend()
plt.show()

# --- VOLATILITY ANALYSIS ---

# rolling 30-day volatility calculation
rolling_vol = returns.rolling(30).std()

# plot volatilities over time
rolling_vol.plot(figsize=(12,6))
plt.title('Rolling 30-Day Volatility of Major FX Pairs')
plt.show()

# compare average volatilities
print("Average volatility during crisis:\n", returns_crisis.std())
print("Average volatility during calm:\n", returns_calm.std())

#download as csv file
vol_stats = pd.DataFrame({
    'Crisis Volatility': returns_crisis.std(),
    'Calm Volatility': returns_calm.std()
})
vol_stats.to_csv('volatility_comparison.csv')

# volatility difference test
levene_results = []
for pair in returns.columns:
    stat, p = levene(returns_crisis[pair].dropna(), returns_calm[pair].dropna())
    interpretation = "Significant" if p < 0.05 else "Not Significant"
    levene_results.append({'pair': pair, 'p_value': p, 'interpretation': interpretation})
    print(f"{pair}: Levene’s test p-value = {p:.4f} → {interpretation} difference in volatility")

#download as csv file
df_levene = pd.DataFrame(levene_results)
df_levene.to_csv('volatility_significance.csv', index=False)

# --- PAIR-LEVEL COMPARISON ---

# define the pair
pair = 'JPY=X'

# price chart
data[pair].plot(figsize=(10, 4), title=f'{pair} Spot Price Over Time')
plt.axvspan('2008-09-01', '2009-06-30', color='red', alpha=0.2, label='2008 Crisis')
plt.axvspan('2020-03-01', '2020-05-31', color='orange', alpha=0.2, label='2020 Crisis')
plt.legend()
plt.ylabel('Price')
plt.show()

# volatility (30-day rolling std of log returns)
returns[pair].rolling(30).std().plot(figsize=(10, 4), title=f'{pair} 30-Day Rolling Volatility')
plt.axvspan('2008-09-01', '2009-06-30', color='red', alpha=0.2)
plt.axvspan('2020-03-01', '2020-05-31', color='orange', alpha=0.2)
plt.ylabel('Volatility')
plt.show()

# return distribution (histograms)
plt.figure(figsize=(10, 4))
sns.histplot(returns_crisis[pair], color='red', label='Crisis', kde=True, stat='density')
sns.histplot(returns_calm[pair], color='blue', label='Calm', kde=True, stat='density')
plt.title(f'{pair} Return Distribution: Crisis vs Calm')
plt.xlabel('Log Return')
plt.legend()
plt.show()

# --- DOWNLOAD OUTPUT ---

with pd.ExcelWriter('table_outputs.xlsx') as writer:
    df_corr_diff.to_excel(writer, index=False, sheet_name='Correlation Changes')
    vol_stats.to_excel(writer, index=True, sheet_name='Volatility')
    df_levene.to_excel(writer, index=False, sheet_name='Levene Test')










