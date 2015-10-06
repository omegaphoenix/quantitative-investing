#===============================================================================
# This code file calculates return for each stock
#===============================================================================

import pandas as pd
import numpy as np
import ReadData

DataFolder = 'D:\Dropbox\CQA 2014\Data'

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
# StockPrices = StockPrices.drop(StockPrices.index[[92330]])
##StockPrices = pd.read_csv(r'E:\Dropbox\CQA 2014\Data\2013stocks.csv',
##                          parse_dates = [1])
##StockPrices.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
##StockBeta = pd.read_excel(r'E:\Dropbox\CQA 2014\Data\Stock Universe 2013.xls')
##StockBeta.columns = ['companyName', 'ticker', 'cqaBeta']
# Now create a new DataFrame to store stock returns
StockReturns = pd.DataFrame()

# Set of all tickers
tickers = StockBeta.ticker
# first_date = StockPrices['date'].min()
# last_date = StockPrices['date'].max()
#
# stock_first = StockPrices[StockPrices['date'] == first_date][['ticker']].reset_index(drop=True)
# stock_last = StockPrices[StockPrices['date'] == last_date][['ticker']].reset_index(drop=True)
# stock_universe = stock_first.merge(stock_last).merge(StockBeta)
# tickers = stock_universe['ticker']

# return on Oct 1st is calculated using opening price of Oct 2nd, say, P2 and Oct 1st, say, P1
# then return on Oct 1st is given by log(P2/P1)
for ticker in tickers:
    print ticker
    tmpData = StockPrices[(StockPrices.ticker == ticker)]
    tmpData = tmpData[['ticker', 'date', 'adj_close']]
    tmpReturn = pd.DataFrame(np.log(tmpData.adj_close/tmpData.adj_close.shift(1)))
    tmpReturn.columns = ['return']
    tmpOutData = pd.concat((tmpData['ticker'][1:].reset_index(drop = True),
                            tmpData['date'][:-1].reset_index(drop = True),
                            tmpData['date'][1:].reset_index(drop = True),
                            tmpReturn[1:].reset_index(drop = True)),
                            axis=1)
    tmpOutData.columns = ['ticker', 'startDate', 'endDate', 'return']
    StockReturns = StockReturns.append(tmpOutData)

##StockReturns = StockReturns[['ticker', 'date', 'return', 'endDate']]
##StockReturns.columns = ['ticker', 'date', 'endDate', 'return']
StockReturns.to_csv(r'%s\StockReturns.csv' % DataFolder,
                    index=False)
