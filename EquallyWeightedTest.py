# This file tests if an equally weighted portfolio can beat SP500

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import ReadData
import functions
# from dateutil import rrule
# from datetime import datetime

DataFolder = 'E:\Dropbox\CQA 2014\Data'
CAPITAL = 1e7
BUILD_DATE = datetime.datetime(2004, 11, 2)

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
minDate = StockPrices['date'].min()
maxDate = StockPrices['date'].max()
tickers = ReadData.ReadTickers(DataFolder)
tickers = tickers[tickers['ticker'] != 'TDS'].reset_index(drop=True)

wTickers = tickers
wTickers['weight'] = 1/np.float64(wTickers.shape[0])

(pfValues, pfRecord) = functions.testPortfolio(StockPrices, StockReturns, SP500Prices, StockBeta, wTickers, BUILD_DATE)

functions.summarize(pfValues, pfRecord, tickers, CAPITAL, BETA_BOUND=0.5)

# findBug = pfRecord[pfRecord['date'] < datetime.datetime(2006,1,1)]
# findBug = findBug[findBug['value']>500000]
#
# tds = pfRecord[(pfRecord['ticker'] == 'TDS') &
#                (pfRecord['date'] > datetime.datetime(2005,5,12)) &
#                (pfRecord['date'] < datetime.datetime(2005,5,20))]
#
# test = StockReturns[StockReturns['return'] > 1]