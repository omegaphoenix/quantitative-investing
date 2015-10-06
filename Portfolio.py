# Test portfolio performance for a static portfolio (no adjustment)
# It's OUTDATED.
import datetime
import pandas as pd
import math
import numpy as np
import ReadData
import functions
import matplotlib.pyplot as plt

DataFolder = 'E:\Dropbox\CQA 2014\Data'
BUILD_DATE = datetime.datetime(2013, 11, 1)
START_TRAIN = datetime.datetime(2010, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
BETA_BOUND = 0.5
MAX_POSITION = 0.05

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockBeta = ReadData.ReadBeta(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
# tickers = ReadData.ReadTickers(DataFolder)
tickers = ReadData.GetTickers(StockPrices, StockBeta, DataFolder)
# Coef = ReadData.ReadCoef(START_TRAIN, END_TRAIN, DataFolder)
# mCoef = Coef[['Mkt-RF', 'SMB', 'HML', 'UMD']].astype(np.float64).as_matrix()
# alphas = Coef[['alphaFF', 'alphaCAPM', 'alphaSP500']].astype(np.float64).as_matrix()
(longTickers, shortTickers, wTickers) = functions.findMax(StockReturns, BUILD_DATE, tickers)
mCqaBeta = pd.merge(StockBeta, tickers).cqaBeta.astype(np.float64).as_matrix()

CAPITAL = 1e7
def intPart(x):
    return math.modf(x)[1]
iniPrices = pd.merge(StockPrices[StockPrices.date == BUILD_DATE][['ticker', 'date', 'adj_close']],
                     tickers, on = 'ticker')
##tickers = iniPrices[['ticker']]
portfolio = pd.concat((tickers, iniPrices[['date']], wTickers[['weight']], iniPrices[['adj_close']]), axis = 1)
portfolio['nShare'] = ((portfolio['weight'] * CAPITAL / portfolio['adj_close']).map(intPart))
portfolio['value'] = portfolio['nShare'] * portfolio['adj_close']
## Recording the initial portfolio
iniPortfolio = portfolio
resCapital = CAPITAL - sum(portfolio['value'])
pfRecord = iniPortfolio
pfValues = pd.DataFrame([{'date' : BUILD_DATE, 'value' : CAPITAL},])

StockPricesTest = pd.merge(StockPrices[StockPrices.date > BUILD_DATE], tickers, on = 'ticker')
StockReturnsTest = pd.merge(StockReturns[StockReturns.endDate > BUILD_DATE], tickers, on = 'ticker')
testDates = StockReturnsTest.endDate.map(lambda t: t.date()).unique()

##testDates = testDates[:5]
for date in testDates:
    print date
    tmpPrice = StockPricesTest[StockPricesTest.date == date][['ticker', 'adj_close']].reset_index(drop=True)
    tmpReturn = StockReturnsTest[StockReturnsTest.endDate == date].reset_index(drop=True)
    tmpData = pd.merge(portfolio.drop(['adj_close', 'date'],1), tmpReturn, on = 'ticker')
    tmpData = pd.merge(tmpData, tmpPrice, on = 'ticker')
    tmpData['value'] = tmpData['value'] * tmpData['return'].map(math.exp)
    tmpValue = sum(tmpData['value']) + resCapital
    tmpData['weight'] = tmpData['value'] / tmpValue
    tmpW = tmpData['weight'].as_matrix()
    # tmpSens = np.dot(tmpW.T, mCoef)
    tmpBeta = np.dot(tmpW, mCqaBeta)
    endDate = tmpReturn['endDate'].iget(0)
    tmpData['date'] = endDate
    tmpData = tmpData[['ticker', 'date', 'weight', 'adj_close', 'nShare', 'value']]
    portfolio = tmpData
    pfRecord = pfRecord.append(tmpData)
    pfValues = pfValues.append(pd.DataFrame([{'date' : endDate,
                                              'value' : tmpValue,
                                              'beta_CQA' : tmpBeta}]))
                                              # 'beta_Mkt-RF' : tmpSens[0],
                                              # 'beta_SMB' : tmpSens[1],
                                              # 'beta_HML' : tmpSens[2],
                                              # 'beta_UMD' : tmpSens[3]},]))

pfValues = pfValues.merge(SP500Prices[['adj_close', 'date']], on = 'date')
pfValues = pfValues.rename(columns = {'adj_close':'SP500'})

def summarize(pfValues):
    plt.figure(1)
    # plt.subplot(321)
    plt.subplot(121)
    plt.ylabel('Account Value vs S&P500')
    plt.grid()
    plt.plot(pfValues.date, pfValues.value, 'r-', pfValues.date, pfValues.SP500 * CAPITAL/pfValues.SP500[0], 'b-')
    plt.axhline(CAPITAL)
    # plt.subplot(322)
    plt.subplot(122)
    plt.ylabel('beta_CQA')
    plt.grid()
    plt.plot(pfValues.date, pfValues.beta_CQA)
    plt.axhline(BETA_BOUND)
    plt.axhline(-BETA_BOUND)
    # plt.subplot(323)
    # plt.ylabel('beta_Mkt-RF')
    # plt.grid()
    # plt.plot(pfValues.date, pfValues['beta_Mkt-RF'])
    # plt.subplot(324)
    # plt.ylabel('beta_SMB')
    # plt.grid()
    # plt.plot(pfValues.date, pfValues.beta_SMB)
    # plt.subplot(325)
    # plt.ylabel('beta_HML')
    # plt.grid()
    # plt.plot(pfValues.date, pfValues.beta_HML)
    # plt.subplot(326)
    # plt.ylabel('beta_UMD')
    # plt.grid()
    # plt.plot(pfValues.date, pfValues.beta_UMD)
    plt.show()

summarize(pfValues)

def viewStocks(pfRecord, tickers):
    for ticker in tickers.ticker:
        print ticker
        tmpData = pfRecord[pfRecord.ticker == ticker]
        plt.plot(tmpData.date, tmpData.value)
    plt.show()

viewStocks(pfRecord, tickers)