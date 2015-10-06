# This file optimizes our portfolio to make it sector neutral

import pandas as pd
import numpy as np
import ReadData
import functions
import datetime
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers, spmatrix
from matplotlib import pylab


data_folder = 'D:\Dropbox\CQA 2014\Data'

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(data_folder)
# StockPrices = StockPrices.drop(StockPrices.index[[92330]])
StockReturns = ReadData.ReadReturn(data_folder)
industry_data = pd.read_csv('%s\industry_data.csv' % data_folder)

first_date = StockPrices['date'].min()
last_date = StockPrices['date'].max()

stock_first = StockPrices[StockPrices['date'] == first_date][['ticker']].reset_index(drop=True)
stock_last = StockPrices[StockPrices['date'] == last_date][['ticker']].reset_index(drop=True)
stock_universe = stock_first.merge(stock_last).merge(StockBeta).merge(industry_data[['ticker', 'sector']])
tickers = stock_universe[['ticker']]

test_dates = np.sort(StockPrices['date'].astype(datetime.datetime).unique())

# build_start_date = datetime.datetime(2011, 11, 1)
# build_end_date = datetime.datetime(2014, 11, 6)
# test_start_date = datetime.datetime(2014, 11, 7)
build_start_date = test_dates[-1000]
build_end_date = test_dates[-250]
test_start_date = test_dates[-249]

test_end_date = last_date
decay = 0.003

prices_build = StockPrices[StockPrices['date'] == build_end_date][['ticker', 'adj_close']]

(Coef, Res) = functions.wReg(StockReturns, CarhartDaily, tickers, decay,
                             build_start_date, build_end_date)

betas = Coef.merge(tickers).rename(columns={'Mkt-RF': 'beta'})[['ticker', 'beta']].reset_index(drop=True)
sortCoef = betas.sort('beta').reset_index(drop=True)
N = sortCoef.shape[0] / 10
long_tickers = sortCoef.head(4*N).tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
short_tickers = sortCoef.tail(4*N)[['ticker']].sort('ticker').reset_index(drop=True)
pfTickers = pd.concat((short_tickers[['ticker']], long_tickers[['ticker']]), axis=0).sort('ticker').reset_index(drop=True)
# def getCovMatrix(tickers):

CarhartSample = CarhartDaily[(CarhartDaily['date'] >= build_start_date) &
                            (CarhartDaily['date'] <= build_end_date)][['Mkt-RF', 'SMB', 'HML', 'UMD']].reset_index(drop=True)
ResSample = Res[pfTickers.ticker]
FactorCov = np.cov(CarhartSample.as_matrix().transpose())
# ResCov = np.cov(ResSample.as_matrix().transpose())
sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers).sort('ticker').drop('ticker', axis=1).as_matrix()

# totalCov = np.dot(np.dot(sens, FactorCov), sens.transpose()) + ResCov
totalCov = np.dot(np.dot(sens, FactorCov), sens.transpose())

BETA_BOUND = 0.3
WEIGHT_BOUND = 0.02
CAPITAL = 1e7
# SECTOR_WEIGHT = 0.05

# BAB, minimize 4-factor covariance matrix, maintain sector neutral
def pfOptimizer_sector(longTickers, shortTickers,
                       Coef, StockBeta, industry_data,
                       BETA_BOUND, WEIGHT_BOUND, SECTOR_WEIGHT):
    longTickers['bWeight'] = 1 # binary weights
    shortTickers['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]), axis=0).sort('ticker').reset_index(drop=True)
    pfTickers = pfTickers.merge(industry_data[['ticker', 'sector']])
    sector_list = pfTickers['sector'].unique()
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

    sector_index = pfTickers[['ticker', 'sector']]
    for sector in sector_list:
        sector_index.loc[:,sector] = 0.0
        sector_index.ix[sector_index['sector'] == sector, sector] = 1.0
    mSector_index = matrix(sector_index.iloc[:, 2:].as_matrix())

    # wTickers = functions.iniWeights(pfTickers, shortTickers, longTickers) # initial weights
    wStart = matrix(functions.iniWeights(pfTickers, longTickers, shortTickers)['weight'])
    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N,1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    # longBounds = matrix(np.ones((shortTickers.shape[0], 1)) * 0.002)
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    # shortBounds = matrix(np.ones((longTickers.shape[0], 1)) * (-0.005))
    A = matrix([[mSector_index], [-mSector_index],
                [mCqaBeta], [-mCqaBeta],
                [longIndex], [-longIndex],
                [shortIndex], [-shortIndex],
                [id], [-id],
                [-mLongIndex], [mShortIndex]]).trans()
    b = matrix([SECTOR_WEIGHT, SECTOR_WEIGHT, SECTOR_WEIGHT, SECTOR_WEIGHT,
                SECTOR_WEIGHT, SECTOR_WEIGHT, SECTOR_WEIGHT, SECTOR_WEIGHT,
                SECTOR_WEIGHT, SECTOR_WEIGHT, SECTOR_WEIGHT, SECTOR_WEIGHT,
                SECTOR_WEIGHT, SECTOR_WEIGHT, SECTOR_WEIGHT, SECTOR_WEIGHT,
                BETA_BOUND, BETA_BOUND,
                1, -0.98,
                -0.98, 1,
                wBounds, wBounds,
                longBounds, shortBounds])

    sol = solvers.lp(-mBeta, A, b)
    w_res = sol['x']
    print 'cqaBeta = %.4f' % np.float64(w_res.trans() * mCqaBeta)[0,0]
    print 'beta = %.4f' % np.float64(w_res.trans() * mBeta)[0,0]
    wTickers = pfTickers
    wTickers['weight'] = w_res
    return wTickers

font_size = 20
color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightblue', 'gold', 'pink']
k = 0
for SECTOR_WEIGHT in [0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001]:
    wTickers = pfOptimizer_sector(long_tickers, short_tickers,
                                  Coef, StockBeta, industry_data,
                                  BETA_BOUND, WEIGHT_BOUND, SECTOR_WEIGHT)
    wTickers = wTickers[abs(wTickers['weight']) > 0.002]
    portfolio = wTickers.merge(prices_build).rename(columns={'adj_close': 'price'})
    portfolio['nShare'] = (portfolio['weight'] * CAPITAL / portfolio['price']).map(functions.intPart)
    pfValues = functions.testPortfolio_simple(StockPrices, StockReturns, SP500Prices, StockBeta, portfolio, test_start_date)
    pfValues['value'] = pfValues['value'] + CAPITAL - pfValues['value'][0]
    plt.plot(pfValues['date'], pfValues['value'],
             linewidth=2.0, color=color_cycle[k],
             label='max sector weight %s' % SECTOR_WEIGHT)
    k += 1

plt.title('Account Value vs Time', size=font_size)
plt.ylabel('Account Value', size=font_size)
plt.xlabel('Date', size=font_size)
plt.grid()
plt.legend(loc=2)
plt.show()