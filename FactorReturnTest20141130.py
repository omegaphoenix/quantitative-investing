# This files tests the Carhart 4 factors return

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import ReadData
import functions
import statsmodels.tsa.stattools as ts
from cvxopt import matrix, solvers, spmatrix
from scipy.optimize import minimize
from matplotlib import pylab

DataFolder = 'D:\Dropbox\CQA 2014\Data'
CAPITAL = 1e7
# START_TRAIN = datetime.datetime(2004, 11, 1)
# END_TRAIN = datetime.datetime(2013, 10, 31)
START = datetime.datetime(2004, 11, 1)
END = datetime.datetime(2014, 11, 5)

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
FactorSample = CarhartDaily[(CarhartDaily.date >= START) &
                            (CarhartDaily.date <= END)].drop('RF', axis=1).reset_index(drop=True)
SP500Sample = SP500Prices[(SP500Prices.date >= START) & (SP500Prices.date <= END)].reset_index(drop=True)
ini_sp500 = SP500Prices[SP500Prices.date < START].tail(1)['adj_close'].iloc[0]

portfolio_mkt = pd.DataFrame()
portfolio_smb = pd.DataFrame()
portfolio_hml = pd.DataFrame()
portfolio_umd = pd.DataFrame()
tmp_mkt = CAPITAL
tmp_smb = CAPITAL
tmp_hml = CAPITAL
tmp_umd = CAPITAL

for row in FactorSample.iterrows():
    date = row[1][0]
    # tmp_mkt *= math.exp(row[1][1]/100)
    # tmp_smb *= math.exp(row[1][2]/100)
    # tmp_hml *= math.exp(row[1][3]/100)
    # tmp_umd *= math.exp(row[1][4]/100)
    tmp_mkt *= (1 + (row[1][1]/100))
    tmp_smb *= (1 + (row[1][2]/100))
    tmp_hml *= (1 + (row[1][3]/100))
    tmp_umd *= (1 + (row[1][4]/100))
    portfolio_mkt = portfolio_mkt.append(pd.DataFrame([{'date': date, 'value': tmp_mkt}]))
    portfolio_smb = portfolio_smb.append(pd.DataFrame([{'date': date, 'value': tmp_smb}]))
    portfolio_hml = portfolio_hml.append(pd.DataFrame([{'date': date, 'value': tmp_hml}]))
    portfolio_umd = portfolio_umd.append(pd.DataFrame([{'date': date, 'value': tmp_umd}]))

plt.plot(portfolio_mkt['date'], portfolio_mkt['value'], label='portfolio_mkt', color='r', linewidth=2.0)
plt.plot(portfolio_smb['date'], portfolio_smb['value'], label='portfolio_smb', color='c', linewidth=2.0)
plt.plot(portfolio_hml['date'], portfolio_hml['value'], label='portfolio_hml', color='g', linewidth=2.0)
plt.plot(portfolio_umd['date'], portfolio_umd['value'], label='portfolio_umd', color='y', linewidth=2.0)
plt.plot(SP500Sample['date'], CAPITAL * SP500Sample['adj_close'] / ini_sp500, label='S&P500', color='b', linewidth=2.0)
plt.legend(bbox_to_anchor=(0.2, 1))
plt.grid()
plt.show()

# factor_cov = np.cov(FactorSample.drop('date', axis=1).as_matrix().transpose())
factor_corr = FactorSample.drop('date', axis=1).corr()

# now test if there's any mean reverting phenomenon of SML/HML/UMD factor

smb_return = FactorSample[['date', 'SMB']].rename( columns={'SMB': 'return'})
hml_return = FactorSample[['date', 'HML']].rename( columns={'HML': 'return'})
umd_return = FactorSample[['date', 'UMD']].rename( columns={'UMD': 'return'})
mkt_return = FactorSample[['date', 'Mkt-RF']].rename( columns={'Mkt-RF': 'return'})

smb_ave_40 = functions.moving_average(smb_return, 40).rename(columns={'ave_return': 'ave_return_40'})
smb_ave_130 = functions.moving_average(smb_return, 130).rename(columns={'ave_return': 'ave_return_130'})
diff_mean = smb_ave_40.merge(smb_ave_130, on='date').reset_index(drop=True)
diff_mean['diff'] = diff_mean['ave_return_40'] - diff_mean['ave_return_130']
threshold = 0.05
plt.plot(diff_mean['date'], diff_mean['diff'], 'r-')
plt.axhline(threshold)
plt.axhline(-threshold)
plt.title('difference between 40 day average and 130 day average')
plt.grid()
plt.show()
# functions.plt_moving_average(smb_return, 'SMB')
# ts.adfuller(smb_return['return'], 1)
# functions.plt_moving_average(hml_return, 'HML')
# ts.adfuller(hml_return['return'], 1)
# functions.plt_moving_average(umd_return, 'UMD')
# ts.adfuller(umd_return['return'], 1)
# functions.plt_moving_average(mkt_return, 'Mkt-RF')
# ts.adfuller(mkt_return['return'], 1)