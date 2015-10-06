import numpy as np
import pandas as pd
import ReadData
import matplotlib.pyplot as plt
import datetime


DataFolder = 'E:\Dropbox\CQA 2014\Data'

StockReturns = ReadData.ReadReturn(DataFolder)
StockReturnsSample = StockReturns[(StockReturns['endDate'] >= datetime.datetime(2010,11,1)) &
                                (StockReturns['endDate'] < datetime.datetime(2013,11,1))].reset_index(drop=True)
StockReturnsTest = StockReturns[(StockReturns['endDate'] >= datetime.datetime(2013,11,1))].reset_index(drop=True)

StockParReturns = pd.DataFrame(StockReturnsSample['endDate'].unique())
StockParReturns.columns = ['endDate']
tickers = pd.DataFrame(StockReturnsSample['ticker'].unique())
tickers.columns = ['ticker']
tickers = tickers.merge(StockReturns[StockReturns['endDate'] == datetime.datetime(2010,11,2)][['ticker']])
tickers = tickers.merge(StockReturns[StockReturns['endDate'] == datetime.datetime(2014,11,6)][['ticker']])
tickers = tickers['ticker']

for ticker in tickers:
    print ticker
    tmpData = StockReturnsSample[StockReturnsSample['ticker'] == ticker].reset_index(drop=True)[['endDate', 'return']].rename( columns = {'return' : ticker})
    # if np.isnan(np.min(tmpData.as_matrix())):
    #     print "undefined return for %s" % ticker
    #     break
    # print type(tmpData)
    # print tmpData
    # StockParReturns[ticker] = tmpData
    StockParReturns = StockParReturns.merge(tmpData)
    # mReturn = StockParReturns.drop('endDate', axis=1).as_matrix()
    # # StockParReturns = pd.concat( (StockParReturns, tmpData), axis = 1)
    # if np.isnan(np.min(mReturn)):
    #     print "undefined return for tickers till %s" % ticker
    #     break

mReturn = StockParReturns.drop('endDate', axis=1).as_matrix().T

CorrMatrix = np.corrcoef(mReturn)
np.fill_diagonal(CorrMatrix, 0)
maxIndex = np.unravel_index(CorrMatrix.argmax(), CorrMatrix.shape)

(ticker1, ticker2) = (tickers[maxIndex[0]], tickers[maxIndex[1]])
corr = np.corrcoef(StockParReturns[[ticker1, ticker2]].as_matrix().T)

#########################################
# Now look at the historical ratios
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
ticker1Data = StockPrices[StockPrices['ticker'] == ticker1][['date', 'adj_close']].rename(columns = {'adj_close':'ticker1Price'})
ticker2Data = StockPrices[StockPrices['ticker'] == ticker2][['date', 'adj_close']].rename(columns = {'adj_close':'ticker2Price'})
comp = ticker1Data.merge(ticker2Data)
comp['ratio'] = comp['ticker1Price'] / comp['ticker2Price']
compTrain = comp[comp['date'] < datetime.datetime(2013,11,1)]

plt.plot(comp['date'], comp['ratio'])
plt.axhline(pd.Series.mean(compTrain['ratio']))
plt.axvline(datetime.datetime(2013,11,1))
plt.show()