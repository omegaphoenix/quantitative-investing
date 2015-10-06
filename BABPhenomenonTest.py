# This file tests the lower-than-expected-return phenomenon for high beta stocks

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import ReadData
import functions

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

decay  = 0.003
Coef = pd.read_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))
# Coef2 = pd.read_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))
betas = Coef[['ticker', 'Mkt-RF']].merge(tickers).rename(columns={'Mkt-RF': 'beta'}).sort('beta')

results = pd.DataFrame()
StockPricesSample = StockPrices[(StockPrices['date'] >= START_TRAIN) &
                                (StockPrices['date'] <= END_TRAIN)].merge(tickers)
for ticker in tickers['ticker']:
    print ticker
    tmpStockPrices = StockPricesSample[StockPricesSample['ticker'] == ticker]
    startPrice = tmpStockPrices[tmpStockPrices['date'] == START_TRAIN]['adj_close'].iloc[0]
    endPrice = tmpStockPrices[tmpStockPrices['date'] == END_TRAIN]['adj_close'].iloc[0]
    tmpReturn = math.log(endPrice/startPrice) * 100 # in percentage
    beta = betas[betas['ticker'] == ticker]['beta'].iloc[0]
    cqaBeta = StockBeta[StockBeta['ticker'] == ticker]['cqaBeta'].iloc[0]
    results = results.append(pd.DataFrame( [{'ticker': ticker,
                                             'beta': beta,
                                             'cqaBeta': cqaBeta,
                                             'return': tmpReturn}]))

plt.figure(1)
plt.subplot(121)
plt.scatter(results['beta'], results['return'])
plt.xlabel('beta')
plt.ylabel('1y return')
plt.title('1y return vs beta')
plt.subplot(122)
plt.scatter(results['cqaBeta'], results['return'])
plt.xlabel('CQA beta')
plt.ylabel('1y return')
plt.title('1y return vs CQA beta')
plt.show()

y = results['return']
# X = sm.add_constant(results['cqaBeta'])
X = results['cqaBeta']
model = sm.OLS(y, X).fit()
model.summary()