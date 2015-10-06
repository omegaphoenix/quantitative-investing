#===============================================================================
# This code file calculates return for each stock (with parallelization)
#===============================================================================

import pandas as pd
import numpy as np
from joblib import Parallel, delayed  
import multiprocessing

StockPrices = pd.read_csv(r'E:\Dropbox\CQA 2014\Data\2013stocks.csv',
                          parse_dates = True)
StockPrices.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
StockBeta = pd.read_excel(r'E:\Dropbox\CQA 2014\Data\Stock Universe 2013.xls')
StockBeta.columns = ['companyName', 'ticker', 'cqaBeta']
# Now create a new DataFrame to store stock returns
StockReturns = pd.DataFrame()

# Set of all tickers
tickers = StockBeta.ticker

# return on Oct 1st is calculated using opening price of Oct 2nd, say, P2 and Oct 1st, say, P1
# then return on Oct 1st is given by log(P2/P1)
def getReturn(ticker, StockReturns):
    print ticker
    tmpData = StockPrices[(StockPrices.ticker == ticker)]
    tmpData = tmpData[['ticker', 'date', 'open']]
    tmpReturn = pd.DataFrame(np.log(tmpData.open/tmpData.open.shift(1)))
    tmpReturn.columns = ['return']
    tmpOutData = pd.concat((tmpData['ticker'][1:].reset_index(drop = True),
                            tmpData['date'][:-1].reset_index(drop = True),
                            tmpReturn[1:].reset_index(drop = True)),
                           axis = 1)
    StockReturns = StockReturns.append(tmpOutData)

NUM_CORES= multiprocessing.cpu_count()
    
Parallel(n_jobs = NUM_CORES)(delayed(getReturn)(ticker) for ticker in tickers)

StockReturns.to_csv(r'E:\Dropbox\CQA 2014\Data\test.csv',
                    index = False)
    

