# optimization test
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
# BUILD_DATE = datetime.datetime(2013, 11, 1)
START_TRAIN = datetime.datetime(2010, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
START = datetime.datetime(2013, 11, 1)
END = datetime.datetime(2014, 11, 5)
BETA_BOUND = 0.45
WEIGHT_BOUND = 0.05
CAPITAL = 1e7

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
SP500Returns = ReadData.GetSP500Return(SP500Prices)
StockReturns = ReadData.ReadReturn(DataFolder)
tickers = ReadData.ReadTickers(DataFolder)

decay  = 0.003
Coef = pd.read_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))
# (StockMax, highMax, longTickers) = functions.findMax(StockReturns, START, tickers)

def pfOptimizer(longTickers, shortTickers, Coef, StockBeta, BETA_BOUND, WEIGHT_BOUND):
    longTickers['bWeight'] = 1 # binary weights
    shortTickers['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]), axis=0).sort('ticker').reset_index(drop=True)

    # sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers).rename(columns={'Mkt-RF': 'beta'})
    # control = matrix([1, 0.2, 0.2, 0.5])
    # scores = matrix(sens[['beta', 'SMB', 'HML', 'UMD']].as_matrix()) * control
    betas = Coef[['ticker', 'Mkt-RF']].merge(pfTickers).rename(columns={'Mkt-RF': 'beta'}).reset_index(drop=True)
    mBeta = matrix(betas['beta'])
    mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])

    longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex,1) > 0.5])]).trans()
    # mLongIndex = matrix(np.diag(tickers.merge(longTickers, how='left').fillna(0)['bWeight']))
    shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex,1) > 0.5])]).trans()
    # mShortIndex = matrix(np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight']))

    # wTickers = functions.iniWeights(pfTickers, shortTickers, longTickers) # initial weights
    wStart = matrix(functions.iniWeights(pfTickers, longTickers, shortTickers)['weight'])
    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N,1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    # longBounds = matrix(np.ones((shortTickers.shape[0], 1)) * 0.002)
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    # shortBounds = matrix(np.ones((longTickers.shape[0], 1)) * (-0.005))
    A = matrix([ [mCqaBeta], [-mCqaBeta],
                 [longIndex], [-longIndex],
                 [shortIndex], [-shortIndex],
                 [id], [-id],
                 [-mLongIndex], [mShortIndex]]).trans()
    b = matrix([BETA_BOUND, BETA_BOUND, 1, -0.98, -0.98, 1, wBounds, wBounds, longBounds, shortBounds])
    # A = matrix([ [longIndex], [-longIndex],
    #              [shortIndex], [-shortIndex],
    #              [id], [-id],
    #              [-mLongIndex], [mShortIndex]]).trans()
    # b = matrix([ 1, -0.98, -0.98, 1, wBounds, wBounds, longBounds, shortBounds])

    # scores = mBeta
    # sol = solvers.lp(-scores, A, b)
    sol = solvers.lp(-mBeta, A, b)
    w_res = sol['x']
    wTickers = pfTickers
    wTickers['weight'] = w_res
    return wTickers

# def sharpeOptimizer(longTickers, shortTickers, Coef, StockBeta, totalCov, thresholds, BETA_BOUND, WEIGHT_BOUND):
#     totalCov = matrix(totalCov)
#     longTickers['bWeight'] = 1 # binary weights
#     shortTickers['bWeight'] = -1
#     pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]), axis=0).sort('ticker').reset_index(drop=True)
#
#     betas = Coef[['ticker', 'Mkt-RF']].merge(pfTickers).rename(columns={'Mkt-RF': 'beta'}).reset_index(drop=True)
#     mBeta = matrix(betas['beta'])
#     mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])
#
#     longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
#     mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
#     mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex,1) > 0.5])]).trans()
#     shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
#     mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
#     mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex,1) > 0.5])]).trans()
#
#     wStart = matrix(functions.iniWeights(pfTickers, longTickers, shortTickers)['weight'])
#     N = pfTickers.shape[0]
#     id = spmatrix(1.0, range(N), range(N))
#     wBounds = matrix(np.ones((N,1)) * WEIGHT_BOUND)
#     longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
#     # longBounds = matrix(np.ones((shortTickers.shape[0], 1)) * 0.002)
#     shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
#     # shortBounds = matrix(np.ones((longTickers.shape[0], 1)) * (-0.005))
#     A = matrix([[-mBeta],
#                 [mCqaBeta], [-mCqaBeta],
#                  [longIndex], [-longIndex],
#                  [shortIndex], [-shortIndex],
#                  [id], [-id],
#                  [-mLongIndex], [mShortIndex]]).trans()
#     sharpes = pd.DataFrame()
#     q = matrix(np.zeros((N,1)))
#     minRatio = 0
#     for threshold in thresholds:
#         print r'threshold for total beta = -%.4f' % threshold
#         b = matrix([threshold, BETA_BOUND, BETA_BOUND, 1, -0.98, -0.98, 1, wBounds, wBounds, longBounds, shortBounds])
#         sol = solvers.qp(totalCov, q, A, b)
#         w_res = sol['x']
#         tmpBeta = np.float64(mBeta.trans() * w_res)[0,0]
#         tmpStd = np.float64(w_res.trans() * totalCov * w_res)[0,0]
#         tmpSharpe = tmpBeta / tmpStd
#         # if tmpSharpe >= minRatio:
#         #     maxSharpe = tmpSharpe
#         #     w_final = w_res
#         sharpes = sharpes.append( pd.DataFrame([ {'threshold': threshold,
#                                                   'beta': tmpBeta,
#                                                   'std': tmpStd,
#                                                   'ratio': tmpSharpe}]))
#     wTickers = pfTickers
#     wTickers['weight'] = w_final
#     return wTickers

# highMaxBeta = StockBeta.merge(highMax)['cqaBeta']
# highMaxBeta.describe()
# highMinBeta = StockBeta.merge(lowMax)['cqaBeta']
# highMinBeta.describe()
# iniBeta = functions.cqaBeta(StockBeta, wTickers)

betas = Coef.merge(tickers).rename(columns={'Mkt-RF': 'beta'})[['ticker', 'beta']].reset_index(drop=True)
sortCoef = betas.sort('beta').reset_index(drop=True)
N = sortCoef.shape[0] / 10
longTickers = sortCoef.head(4*N).tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
shortTickers = sortCoef.tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)

wTickers = pfOptimizer(longTickers, shortTickers, Coef, StockBeta, BETA_BOUND, WEIGHT_BOUND)
wTickers = pfOptimizer(longTickers, shortTickers, Coef, StockBeta, 0.4, 0.01)
wTickers2 = wTickers[np.abs(wTickers['weight']) > 0.001]

# thresholds = 0.5 - np.float64(range(11)) * 0.025
thresholds = [0.3, 0.295, 0.29, 0.285, 0.28, 0.275, 0.27, 0.265, 0.26]

(pfValues, pfRecord) = functions.testPortfolio(StockPrices, StockReturns, SP500Prices,
                                               StockBeta, wTickers, START)

functions.summarize(pfValues, pfRecord, wTickers[['ticker']], CAPITAL, BETA_BOUND)

pfValues['return'] = np.log(pfValues['value']/pfValues['value'].shift(1))

Sharpe = pd.DataFrame()
for date in pfValues.date[20:]:
    totalReturn = np.mean(pfValues[pfValues['date'] <= date][1:]['return'])
    totalStd = np.std(pfValues[pfValues['date'] <= date][1:]['return'])
    tmpSharpe = pd.DataFrame([{'date': date,
                               'return': totalReturn,
                               'std': totalStd,
                               'ratio': totalReturn/totalStd}])
    Sharpe = Sharpe.append(tmpSharpe)

plt.plot(Sharpe['date'], Sharpe['ratio'])
plt.show()

SP500ReturnsSample = SP500Returns[SP500Returns['endDate'] >= min(pfValues['date'])]
SP500Sharpe = np.mean(SP500ReturnsSample['SP500Return'])/np.std(SP500ReturnsSample['SP500Return'])