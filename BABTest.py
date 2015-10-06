# This file tests if high beta stocks will have low returns

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import ReadData
import functions
from cvxopt import matrix, solvers, spmatrix
from scipy.optimize import minimize

DataFolder = 'E:\Dropbox\CQA 2014\Data'
CAPITAL = 1e7
# BUILD_DATE = datetime.datetime(2013, 11, 1)
START_TRAIN = datetime.datetime(2010, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
START = datetime.datetime(2013, 11, 1)
END = datetime.datetime(2014, 11, 5)
BETA_BOUND = 0.45
WEIGHT_BOUND = 0.05

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
tickers = ReadData.ReadTickers(DataFolder)
(StockMax, highMax, lowMax) = functions.findMax(StockReturns, START, tickers)

decay  = 0.003
Coef = pd.read_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))
betas = Coef[['ticker', 'Mkt-RF']].merge(tickers).rename(columns={'Mkt-RF': 'beta'}).sort('beta')
maxes = StockMax[['ticker', 'MAX']].merge(tickers).sort('MAX')
N = betas.shape[0]/10

def groupPerformance(StockPrices, StockReturns, SP500Prices, StockBeta, betas, START):
    factor = betas.columns.values[1]
    betas = betas.sort(factor)
    pfValuesAll = pd.DataFrame()
    for n in range(10):
        wTickers = betas.head(N * (n+1)).tail(N)[['ticker']].reset_index(drop=True)
        wTickers['weight'] = 1/np.float64(N)
        (pfValues, pfRecord) = functions.testPortfolio(StockPrices, StockReturns, SP500Prices, StockBeta, wTickers, START)
        pfValues['group'] = n+1
        pfValuesAll = pfValuesAll.append(pfValues)

    color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightblue', 'gold', 'pink']
    for n in range(10):
        pfValues = pfValuesAll[pfValuesAll['group'] == n+1 ]
        tmpLabel = r'group %d' % (n+1)
        plt.plot(pfValues['date'], pfValues['value'], label=tmpLabel, color=color_cycle[n], linewidth=3.0)
        # plt.legend('beta group %d' % (n+1))
    plt.ylabel('Portfolio Value')
    plt.legend(bbox_to_anchor=(0.2, 1))
    plt.show()
    return pfValuesAll

betaPf = groupPerformance(StockPrices, StockReturns, SP500Prices, StockBeta, betas, START)
betaResult = betaPf[betaPf['date'] == max(betaPf['date'])]
plt.plot(betaResult['group'], (betaResult['value']/CAPITAL)-1)
plt.ylabel('1y Return')
plt.xlabel('Beta Group')
plt.title('1y Return of various Beta Groups')
plt.grid()
plt.show()

maxPf = groupPerformance(StockPrices, StockReturns, SP500Prices, StockBeta, maxes, START)
maxResult = maxPf[maxPf['date'] == max(maxPf['date'])]
plt.plot(maxResult['group'], (maxResult['value']/CAPITAL)-1)
plt.ylabel('1y Return')
plt.xlabel('MAX Group')
plt.title('1y Return of various MAX Groups')
plt.grid()
plt.show()

