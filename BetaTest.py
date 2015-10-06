import math
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import datetime
import ReadData
from scipy.optimize import minimize

## We need to register datetime64 as a recognizable format in matplotlib first
matplotlib.units.registry[np.datetime64] = pd.tseries.converter.DatetimeConverter()

START_TRAIN = datetime.datetime(2010, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
START = datetime.datetime(2013, 11, 1)
END = datetime.datetime(2014, 10, 31)

DataFolder = 'D:\Dropbox\CQA 2014\Data'
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
SP500Returns = ReadData.GetSP500Return(SP500Prices)
print "Finished reading data..."
RF = CarhartDaily[['date','RF']]

##############################################################################
## Compare returns of Mkt-RF and SP500Returns-RF
##returnComp = SP500Returns[['endDate','SP500Return']].merge(CarhartDaily[['date','Mkt-RF','RF']],
##                                                   left_on = 'endDate', right_on = 'date', how = 'inner')
##returnComp['SP500-RF'] = returnComp['SP500Return'] * 100 - returnComp['RF']
##returnCompRecent = returnComp.tail(200)
##
##def returnPlot(returns):
##    returns = returns.sort('Mkt-RF')
##    plt.figure(1)
##    plt.subplot(211)
##    plt.scatter(returns['Mkt-RF'], returns['SP500-RF'])
##    plt.plot(range(-3, 3), range(-3, 3))
##    plt.subplot(212)
##    plt.plot(returns['Mkt-RF'])
##    plt.plot(returns['SP500-RF'])
##    plt.show()
##
##plt.figure(1)
##plt.subplot(211)
##returnPlot(returnComp)
##plt.subplot(212)
##returnPlot(returnCompRecent)
##plt.show()

## Use AAPL as an example to test how beta_CQA is calculated (intercept, time_range)
##aapl = StockReturns[StockReturns.ticker == 'AAPL'].merge(CarhartDaily[['date', 'Mkt-RF', 'RF']],
##                                                         left_on = 'endDate', right_on = 'date').drop('date',1)
##aapl = aapl.merge(SP500Returns.drop('startDate', 1), on = 'endDate')
##aapl['SP500-RF'] = aapl['SP500Return']*100 - aapl['RF']
##aapl['return-RF'] = aapl['return']*100 - aapl['RF']
##aapl3y = aapl.tail(252 * 3)
##
##beta_corr = np.corrcoef(aapl3y['return-RF'], aapl3y['Mkt-RF'])[1,0]
##beta_corr2 = np.corrcoef(aapl3y['return-RF'], aapl3y['SP500-RF'])[1,0]
##y = aapl3y['return-RF']
##X = aapl3y['SP500-RF']
##
##def fun1(y, X):
##    model = sm.OLS(y, X).fit()
##    print aaplModel.summary()
##fun1(y, X) # this returns 0.84, which is very close to beta_CQA = 0.83
##fun1(y, sm.add_constant(X))


##############################################################################
# Calculate or Read regression coefficients, depending on your own needs
# Coef = ReadData.ReadCoef(START_TRAIN, END_TRAIN, DataFolder)
# Coef2 = ReadData.ReadCoef(datetime.datetime(2011,11,1),
#                          datetime.datetime(2014,10,31), DataFolder)
Coef = ReadData.GetCoef(START_TRAIN, END_TRAIN, StockReturns, CarhartDaily, SP500Returns, DataFolder)
# Coef2 = ReadData.GetCoef(datetime.datetime(2011,11,1),
#                          datetime.datetime(2014,10,31),
#                          StockReturns, CarhartDaily, SP500Returns, DataFolder)

avaTickers = pd.merge(Coef[['ticker']], StockBeta[ np.isnan(StockBeta.cqaBeta) == False][['ticker']])
avaTickers = pd.merge(avaTickers, StockPrices[StockPrices.date == START_TRAIN][['ticker']], on = 'ticker')
avaTickers = pd.merge(avaTickers, StockPrices[StockPrices.date == END_TRAIN][['ticker']], on = 'ticker')
avaTickers = pd.merge(avaTickers, StockPrices[StockPrices.date == START][['ticker']], on = 'ticker')
avaTickers = pd.merge(avaTickers, StockPrices[StockPrices.date == END][['ticker']], on = 'ticker')
Coef = Coef.merge(avaTickers, on = 'ticker')

betaComp = Coef[['ticker', 'Mkt-RF', 'Mkt-RF_only', 'SP500-RF']].merge(StockBeta[['ticker','cqaBeta']], on = 'ticker')
# betaComp2 = Coef2[['ticker', 'Mkt-RF', 'Mkt-RF_only', 'SP500-RF']].merge(StockBeta[['ticker','cqaBeta']]).merge(avaTickers)
##betaComp = betaComp.sort('cqaBeta')

def betaCompPlot(b):
    print 'Summary of cqaBeta-beta_Mkt, CAPM'
    print (b['cqaBeta'] - b['Mkt-RF_only']).describe()
    print 'Summary of cqaBeta-SP500_only, CAPM'
    print (b['cqaBeta'] - b['SP500-RF']).describe()
    print 'Summary of cqaBeta-beta_Mkt, Fama-French'
    print (b['cqaBeta'] - b['Mkt-RF']).describe()
    plt.figure(1)
    plt.subplot(221)
    plt.scatter(b['cqaBeta'], b['Mkt-RF_only'])
    plt.plot(range(4), range(4), 'r-')
    plt.grid()
    plt.title('regression coefficients on Mkt-RF only vs beta_CQA')
    plt.subplot(222)
    plt.scatter(b['cqaBeta'], b['SP500-RF'])
    plt.plot(range(4), range(4), 'r-')
    plt.grid()
    plt.title('regression coefficients on SP500 only vs beta_CQA')
    plt.subplot(223)
    plt.scatter(b['cqaBeta'], b['Mkt-RF'])
    plt.plot(range(4), range(4), 'r-')
    plt.grid()
    plt.title('regression coefficients FamaFrench on Mkt-RF vs beta_CQA')
    plt.show()

# betaCompPlot(betaComp)
# betaCompPlot(betaComp2)

# Now compare historical beta and future correlation
stock_returns_sample = StockReturns[(StockReturns['endDate'] >= START_TRAIN) &
                                    (StockReturns['endDate'] <= END_TRAIN)]
corr_list = pd.DataFrame()
for ticker in avaTickers['ticker']:
    print ticker
    tmp_return = stock_returns_sample[stock_returns_sample['ticker'] == ticker][['endDate', 'return']]
    tmp_return = tmp_return.merge(SP500Returns[['endDate', 'SP500Return']], on='endDate')
    tmp_corr = tmp_return['return'].corr(tmp_return['SP500Return'])
    corr_list = corr_list.append(pd.DataFrame([{'ticker': ticker,
                                    'corr': tmp_corr}]))

beta_corr_comp = betaComp.merge(corr_list, on='ticker')[['ticker', 'Mkt-RF', 'corr']]
plt.figure()
matplotlib.rc('xtick', labelsize=36)
matplotlib.rc('ytick', labelsize=36)
plt.scatter(beta_corr_comp['Mkt-RF'], beta_corr_comp['corr'], marker='x')
plt.legend(loc=3, prop={'size': 36})
plt.grid()
plt.show()


