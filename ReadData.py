#===============================================================================
# This code defines function to read data
#===============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
import datetime
import os.path

# DataFolder = 'E:\Dropbox\CQA 2014\Data'

# The following function reads stockPrices, SP500Prices, Fama-French-Carhart factor daily return, cqaBeta
def ReadAll(DataFolder):
    ## We need to register datetime64 as a recognizable format in matplotlib first
    matplotlib.units.registry[np.datetime64] = pd.tseries.converter.DatetimeConverter()

    StockPrices = pd.read_csv(r'%s\prices_20150206.csv' % DataFolder,
                              parse_dates=[1])
    StockPrices.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
    # StockPrices = StockPrices[StockPrices['date'] < datetime.datetime(2014, 11, 1)]
    # # NovPrices = pd.read_csv(r'%s\prices.csv' % DataFolder,
    # #                           parse_dates=[1])
    # NovPrices = pd.read_csv(r'%s\PortfolioPerformance.csv' % DataFolder,
    #                         parse_dates=[1])
    # NovPrices.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
    # # NovPrices = NovPrices.sort(columns=['ticker', 'date'])
    # StockPrices = StockPrices.append(NovPrices)

    SP500Prices = pd.read_csv(r'%s\sp500.csv' % DataFolder,
                              parse_dates = [0])
    SP500Prices.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
    SP500Prices = SP500Prices.sort('date').reset_index(drop = True)
    # StockReturns = pd.read_csv(r'%s\StockReturns.csv' % DataFolder,
    #                           parse_dates = [1,2])
    # StockReturns.columns= ['ticker', 'startDate', 'endDate', 'return']
    StockBeta = ReadBeta(DataFolder)
    FamaFrenchDaily = pd.read_table(r'%s\F-F_Research_Data_Factors_daily.txt' % DataFolder,
                                    header=0,
                                    delim_whitespace=True,
                                    skiprows=range(5) + [23410, 23411, 23412],
                                    parse_dates=[0])
    FamaFrenchDaily.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    MomentumDaily = pd.read_table(r'%s\F-F_Momentum_Factor_daily.txt' % DataFolder,
                                  header=0,
                                  delim_whitespace=True,
                                  skiprows=range(14) + [23318, 23319, 23320],
                                  parse_dates=[0])
    MomentumDaily.columns = ['date', 'UMD']
    CarhartDaily = pd.merge(FamaFrenchDaily, MomentumDaily,
                            on='date')

    return StockPrices, SP500Prices, CarhartDaily, StockBeta

# Since calculating return for each stock takes some time, we can read it from disk instead.
def ReadReturn(DataFolder):
    # The 2nd, 3rd columns are dates so we need to parse them.
    StockReturns = pd.read_csv(r'%s\StockReturns.csv' % DataFolder,
                              parse_dates=[1,2])
    StockReturns.columns= ['ticker', 'startDate', 'endDate', 'return']
    return StockReturns

def ReadCoef(START_TRAIN, END_TRAIN, DataFolder):
    filename = r'%s\Coef_%s_%s.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date())
    Coef = pd.read_csv(filename,
                   dtype = {'Mkt-RF' : np.float64,
                            'SMB' : np.float64,
                            'HML' : np.float64,
                            'UMD' : np.float64},
                              parse_dates = [1])
    Coef = Coef.rename(columns = {'symbol' : 'ticker'})
    # For some unknown reason we need to deal with 'SMB' separately
    Coef[['SMB']] = Coef[['SMB']].astype(np.float64)
    Coef[['Mkt-RF']] = Coef[['Mkt-RF']].astype(np.float64)
    return Coef

def ReadBeta(DataFolder):
    StockBeta = pd.read_excel(r'%s\Russell 1000 with betas as of 10 15 2014.xls' % DataFolder,
                              skiprows = [1037, 1038, 1039])
    StockBeta.columns = ['ticker', 'companyName', 'cqaBeta']
    StockBeta = StockBeta[np.isnan(StockBeta.cqaBeta) == False]
    return StockBeta

# Here we run (OLS) regression of stock returns over Fama-French-Carhart factors
# Be careful since Fama-French-Carhart factor return is in percentage
# In each for loop, we run 3 regression to see different beta estimation. They tend to be similar to each other.
def GetCoef(start_train, end_train, StockReturns, CarhartDaily, SP500Returns, DataFolder):
    if os.path.isfile(r'%s\Coef_%s_%s.csv' % (DataFolder, start_train.date(), end_train.date())):
        Coef = pd.read_csv(r'%s\Coef_%s_%s.csv' % (DataFolder, start_train.date(), end_train.date()))
        return Coef
    else:
        Coef = pd.DataFrame()
        for ticker in StockReturns.ticker.unique():
            print "Getting regression coefficient for %s" % ticker
            tmpReturn = StockReturns[(StockReturns.ticker == ticker)]
            if not tmpReturn.empty:
                tmpData = tmpReturn.merge(CarhartDaily, left_on = 'endDate', right_on = 'date')
                tmpData = tmpData.merge(SP500Returns, on = 'endDate')
                tmpData['SP500-RF'] = tmpData['SP500Return']*100 - tmpData['RF']
                y = tmpData['return']*100 - tmpData['RF']
                X1 = tmpData[['Mkt-RF', 'SMB', 'HML', 'UMD']]
                X2 = tmpData[['Mkt-RF']]
                X3 = tmpData[['SP500-RF']]
                X1 = sm.add_constant(X1)
                X2 = sm.add_constant(X2)
                X3 = sm.add_constant(X3)
                model1 = sm.OLS(y, X1).fit()
                model2 = sm.OLS(y, X2).fit()
                model3 = sm.OLS(y, X3).fit()
                tmpDF1 = pd.DataFrame(model1.params).T
                tmpDF1.rename( columns = {'const' : 'alphaFF'}, inplace = True)
                tmpDF2 = pd.DataFrame(model2.params).T
                tmpDF2.rename( columns = {'const' : 'alphaCAPM', 'Mkt-RF' : 'Mkt-RF_only'}, inplace = True)
                tmpDF3 = pd.DataFrame(model3.params).T
                tmpDF3.rename( columns = {'const' : 'alphaSP500'}, inplace = True )
                tmpDF = pd.concat((tmpDF1, tmpDF2, tmpDF3), axis = 1)
                tmpDF['ticker'] = ticker
                Coef = Coef.append(tmpDF)
        Coef.to_csv(r'%s\Coef_%s_%s.csv' % (DataFolder, start_train.date(), end_train.date()), index = False)
        print 'Finished saving regression coefficients to: %s\Coef_%s_%s.csv' % (DataFolder, start_train.date(), end_train.date())
        return Coef

# The following function calculates returns from the daily SP500 adj_close price.
# Note that the calculated return is in absolute value (not in percentage)
def GetSP500Return(SP500Prices):
    SP500Returns = pd.concat((SP500Prices['date'][:-1].reset_index(drop = True),
                              SP500Prices['date'][1:].reset_index(drop = True),
                              (np.log(SP500Prices.adj_close/SP500Prices.adj_close.shift(1)))[1:].reset_index(drop = True)),
                              axis = 1)
    SP500Returns.columns = ['startDate', 'endDate', 'SP500Return']
    return SP500Returns

# The following function calculates returns from the daily prices for each stock.
# The price used here is adj_price
# Note that the calculated return is in absolute value (not in percentage)
def GetReturn(prices, DataFolder):
    tickers = prices['ticker'].unique()
    returns = pd.DataFrame()
    for ticker in tickers:
        print ticker
        tmpData = prices[(prices.ticker == ticker)]
        tmpData = tmpData[['ticker', 'date', 'adj_close']]
        tmpReturn = pd.DataFrame(np.log(tmpData.adj_close/tmpData.adj_close.shift(1)))
        tmpReturn.columns = ['return']
        tmpOutData = pd.concat((tmpData['ticker'][1:].reset_index(drop = True),
                                tmpData['date'][:-1].reset_index(drop = True),
                                tmpData['date'][1:].reset_index(drop = True),
                                tmpReturn[1:].reset_index(drop = True)),
                               axis = 1)
        tmpOutData.columns = ['ticker', 'startDate', 'endDate', 'return']
        returns = returns.append(tmpOutData)
    returns = returns.reset_index(drop=True)
    returns.to_csv(r'%s\StockReturns.csv' % DataFolder, index=False)
    return returns

def GetTickers(StockPrices, StockBeta, DataFolder):
    tickers = StockBeta[['ticker']]
    startTickers = pd.DataFrame(StockPrices[StockPrices['date'] == datetime.datetime(2004, 11, 2)]['ticker'].unique(),
                                columns=['ticker'])
    endTickers = pd.DataFrame(StockPrices[StockPrices['date'] == datetime.datetime(2014, 11, 3)]['ticker'].unique(),
                                columns=['ticker'])
    tickers = tickers.merge(startTickers)
    tickers = tickers.merge(endTickers)
    tickers.to_csv(r'%s/tickers.csv' % DataFolder, index=False)
    return tickers

def ReadTickers(DataFolder):
    tickers = pd.read_csv(r'%s/tickers.csv' % DataFolder)
    return tickers