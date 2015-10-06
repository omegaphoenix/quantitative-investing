# This file tests the performance of each beta group

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
StockReturns = ReadData.ReadReturn(data_folder)
industry_data = pd.read_csv('%s\industry_data.csv' % data_folder)

first_date = StockPrices['date'].min()
last_date = StockPrices['date'].max()

stock_first = StockPrices[StockPrices['date'] == first_date][['ticker']].reset_index(drop=True)
stock_last = StockPrices[StockPrices['date'] == last_date][['ticker']].reset_index(drop=True)
stock_universe = stock_first.merge(stock_last).merge(StockBeta).merge(industry_data[['ticker', 'sector']])
tickers = stock_universe[['ticker']]

test_dates = np.sort(StockPrices['date'].astype(datetime.datetime).unique())
build_start_date = datetime.datetime(2011, 11, 1)
build_end_date = datetime.datetime(2014, 11, 7)
test_start_date = datetime.datetime(2014, 11, 10)

test_end_date = last_date
decay = 0.003

prices_build = StockPrices[StockPrices['date'] == build_end_date][['ticker', 'adj_close']]

(Coef, Res) = functions.wReg(StockReturns, CarhartDaily, tickers, decay,
                             build_start_date, build_end_date)
CAPITAL = 1e7
betas = Coef.merge(tickers).rename(columns={'Mkt-RF': 'beta'})[['ticker', 'beta']].reset_index(drop=True)
sortCoef = betas.sort('beta').reset_index(drop=True)
N = sortCoef.shape[0] / 10

font_size = 20
color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightblue', 'gold', 'pink']
pfValuesAll = pd.DataFrame()
for n in range(10):
    tmpData = sortCoef.head(N * (n + 1)).tail(N)
    mean_beta = tmpData['beta'].mean()
    wTickers = tmpData[['ticker']].reset_index(drop=True)
    wTickers['weight'] = 1 / np.float64(N)
    portfolio = wTickers.merge(prices_build).rename(columns={'adj_close': 'price'})
    portfolio['nShare'] = (CAPITAL * portfolio['weight'] / portfolio['price']).map(functions.intPart)
    pfValues = functions.testPortfolio_simple(StockPrices, StockReturns, SP500Prices, StockBeta,
                                              portfolio, test_start_date)
    pfValues['beta group'] = n
    pfValues['mean beta'] = mean_beta
    pfValues['value'] = pfValues['value'] + CAPITAL - pfValues['value'][0]
    plt.plot(pfValues['date'], pfValues['value'],
             linewidth=2.0, color=color_cycle[n],
             label='mean beta %s' % mean_beta)
    pfValuesAll = pfValuesAll.append(pfValues)


plt.title('Performance of Beta Groups', size=font_size)
plt.ylabel('Account Value', size=font_size)
plt.xlabel('Date', size=font_size)
plt.grid()
plt.legend(loc=3)
plt.show()


# now test the performance since Jan 16th, 2015
pfValuesAll_late_Jan = pfValuesAll[pfValuesAll['date'] >= datetime.datetime(2015, 1, 16)]
for n in range(10):
    pfValues = pfValuesAll_late_Jan[pfValuesAll_late_Jan['beta group'] == n].reset_index(drop=True)
    pfValues['value'] /= pfValues['value'].iloc[0]
    mean_beta = pfValues['mean beta'].iloc[0]
    plt.plot(pfValues['date'], pfValues['value'],
             linewidth=2.0, color=color_cycle[n],
             label='mean beta %s' % mean_beta)
plt.title('Performance of Beta Groups (since Jan 16, 2015)', size=font_size)
plt.ylabel('Account Value', size=font_size)
plt.xlabel('Date', size=font_size)
plt.grid()
plt.legend(loc=2)
plt.show()

longPosition = pfValuesAll_late_Jan[pfValuesAll_late_Jan['beta group'] == 3]
shortPosition = pfValuesAll_late_Jan[pfValuesAll_late_Jan['beta group'] == 7]
longPosition['return'] = np.log(longPosition['value']/longPosition['value'].shift(1))*100
shortPosition['return'] = np.log(shortPosition['value']/shortPosition['value'].shift(1))*100


def long_short_return_bar():
    n_groups = longPosition.shape[0]-1
    font_size = 15
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, longPosition['return'][1:], bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Long Position Daily Return')
    rects2 = plt.bar(index + bar_width, shortPosition['return'][1:], bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='Short Position Daily Return')
    plt.xlabel('Date', size=font_size)
    plt.ylabel('Return (in percentage)', size=font_size)
    plt.title('Long/Shoart Daily Return since Jan 16, 2015', size=font_size)
    plt.xticks(index + bar_width,
               map(lambda d: d.date(), list(longPosition['date'][1:])))
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

long_short_return_bar()