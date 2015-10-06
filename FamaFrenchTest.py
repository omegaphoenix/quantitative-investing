#===============================================================================
# This code implements the Fama-French 4-factor model
# It's OUTDATED.
#===============================================================================

import math
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import datetime
import ReadData
# import cvxopt
# import pymatlab
##import pylab
##from statsmodels.sandbox.descstats import descstats
##from pylab import figure, show
from scipy.optimize import minimize

DISK = 'E'

## We need to register datetime64 as a recognizable format in matplotlib first
matplotlib.units.registry[np.datetime64] = pd.tseries.converter.DatetimeConverter()

## Read data
(StockPrices, StockReturns, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DISK)
print "Finished reading data..."
SP500Returns = ReadData.GetSP500Return(SP500Prices)

## For training purpose we only use the data before 2013/11/01 (start date of testing period)
START_TRAIN = datetime.datetime(2010, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
START = datetime.datetime(2013, 11, 1)
END = datetime.datetime(2014, 10, 31)
CarhartDailyTrain = CarhartDaily[
    (CarhartDaily.date >= START_TRAIN) &
    (CarhartDaily.date < START)]

## Delete stock tickers without beta value
## We need to make sure all stocks exist on the whole testing period
tickers = StockBeta[ np.isnan(StockBeta.cqaBeta) == False][['ticker']]
avaTickers = pd.merge(tickers, StockPrices[StockPrices.date == START][['ticker']], on = 'ticker')
avaTickers = pd.merge(avaTickers, StockPrices[StockPrices.date == END][['ticker']], on = 'ticker')
##tickers = tickers[:50]
##tickers = ['A']

##################################################################
## Run this part if you already have the return file, otherwise calculate it
Coef = ReadData.ReadCoef(START_TRAIN, END_TRAIN, DISK)
##################################################################
Coef = pd.merge(Coef, avaTickers, on = 'ticker')

print "Now we have all the regression coefficients..."
#### test optimization, using only 5 stocks
##avaTickers = avaTickers.head(5)
mCqaBeta = pd.merge(StockBeta, avaTickers).cqaBeta.astype(np.float64).as_matrix()
np.savetxt(r'%s:\Dropbox\CQA 2014\Data\cqaBeta.csv' % DISK, mCqaBeta, delimiter=",")
NUM_STOCKS = avaTickers.shape[0]

mCoef = Coef[['Mkt-RF', 'SMB', 'HML', 'UMD']].astype(np.float64).as_matrix()
alphas = Coef[['alphaFF', 'alphaCAPM', 'alphaSP500']].astype(np.float64).as_matrix()
## Now build our portfolio based on the regression coefficients
l_SMB = 0
l_HML = 0
l_UMD = 0
l_alphaFF = 1
l_alphaCAPM = 1
l_alphaSP500 = 1
betaBound = 0.45
BETA_BOUND = 0.5
MAX_POSITION = 0.05
wBounds = ((-MAX_POSITION, MAX_POSITION),) * NUM_STOCKS
control = np.array([1, l_SMB, l_HML, l_UMD, l_alphaFF, l_alphaCAPM, l_alphaSP500])
stockRisk = np.dot(np.concatenate((mCoef, alphas), axis=1), control.T)
np.savetxt(r'%s:\Dropbox\CQA 2014\Data\stockRisk.csv' % DISK, stockRisk, delimiter=",")

## w is a column vector (array), length = number of stocks in universe
def totalRisk(w, mCoef = mCoef, alphas = alphas,
              l_SMB = l_SMB, l_HML = l_HML, l_UMD = l_UMD,
              l_alphaFF = l_alphaFF, l_alphaCAPM = l_alphaCAPM, l_alphaSP500 = l_alphaSP500):
    sens = np.concatenate((np.dot(w.T, mCoef), np.dot(w.T, alphas)), axis = 1)
    print 'Sensitivity to each factor:'
    print sens
    print 'totalCqaBeta = %.5f' % totalCqaBeta(w)
    control = np.array([1, l_SMB, l_HML, l_UMD, l_alphaFF, l_alphaCAPM, l_alphaSP500])
    res = -np.dot(control, sens.T)
    if isinstance(res, np.ndarray):
        res = res[0]
    return res

def totalCqaBeta(w, mCqaBeta = mCqaBeta):
    res = np.dot(mCqaBeta, w)
    if isinstance(res, np.ndarray):
        res = res[0]
    return abs(res)


## Constraint on cqaBeta
def max0(x):
    return max(x,0)
def min0(x):
    return min(x,0)
            
vMax0 = np.vectorize(max0, otypes=[np.float64])
vMin0 = np.vectorize(min0, otypes=[np.float64])

cons = ({'type': 'ineq', 'fun': lambda w: (betaBound - totalCqaBeta(w))},
        {'type': 'ineq', 'fun': lambda w: (0.01 - (np.sum(w)))},
        {'type': 'ineq', 'fun': lambda w: (0.01 + (np.sum(w)))},
        {'type': 'ineq', 'fun': lambda w: (1 - (np.sum(vMax0(w))))},
        {'type': 'ineq', 'fun': lambda w: (1 + (np.sum(vMin0(w))))})
# def f1(w):
#     return betaBound - totalCqaBeta(w)
# def f2(w):
#     return 0.01 - abs(np.sum(w))
# def f3(w):
#     return 1 - abs(np.sum(vMax0(w)))
# def f4(w):
#     return 1 - abs(np.sum(vMin0(w)))
# def f5(w):
#     return 0.05 - max(w)
# def f6(w):
#     return 0.05 + min(w)
# def f1to4(w):
#     return np.array([f1(w), f2(w), f3(w), f4(w)])
# cons2 = [{'type': 'ineq', 'fun': f1},
#         {'type': 'ineq', 'fun': f2},
#         {'type': 'ineq', 'fun': f3},
#         {'type': 'ineq', 'fun': f4},
#          {'type': 'ineq', 'fun': f5},
#          {'type': 'ineq', 'fun': f6}]
##cons2 = [{'type': 'ineq', 'fun': lambda w: 0.05 - max(w)[0]},
##         {'type': 'ineq', 'fun': lambda w: 0.05 + min(w)[0]}]

part1 = NUM_STOCKS/2
part2 = NUM_STOCKS - NUM_STOCKS/2
w0 = np.concatenate((np.ones((part1,1)) * 1/part1,
                     np.ones((part2,1)) * (-1)/part2), axis = 0).astype(np.float64)

raw_input("Waiting for MATLAB result...Press Enter to continue...")

# ##w_raw = minimize(totalRisk, w0, bounds = wBounds, constraints = cons, method='SLSQP', options={'maxiter' : 1, 'disp': True})
# w_raw = minimize(totalRisk, w0, bounds = wBounds, constraints = cons, method='SLSQP', options={'disp': True})
# ##w_raw = minimize(totalRisk, w0, constraints = cons, method = 'COBYLA', options={'maxiter' : 10000, 'disp': True})
# ##w_raw = sp.optimize.fmin_cobyla(totalRisk, w0, [f1, f2, f3, f4, f5, f6])
# # w_raw = sp.optimize.fmin_slsqp(totalRisk, w0, f_ieqcons = f1to4, bounds = wBounds)
# w_res = w_raw.x
w_res = pd.read_csv(r'%s:\Dropbox\CQA 2014\Data\weights.csv' % DISK, header = None).astype(np.float64).as_matrix()

print "total CQA beta = %.5f." % totalCqaBeta(w_res)
print "total beta_Mkt-RF = %.5f." % totalCqaBeta(w_res)
print "Successfully initialized the weights..."
## Now we get a portfolio at end of 2013/10/31. Let's see its performance in the next year.
CAPITAL = 1e7
def intPart(x):
    return math.modf(x)[1]
iniPrices = pd.merge(StockPrices[StockPrices.date == START][['ticker', 'date', 'adj_close']],
                     avaTickers, on = 'ticker')
##avaTickers = iniPrices[['ticker']]
portfolio = pd.concat((avaTickers, iniPrices[['date']],
                       pd.DataFrame(w_res, columns = ['weight']),
                       iniPrices[['adj_close']]), axis = 1)
portfolio['nShare'] = ((portfolio['weight'] * CAPITAL / portfolio['adj_close']).map(intPart))
portfolio['value'] = portfolio['nShare'] * portfolio['adj_close']
## Recording the initial portfolio
iniPortfolio = portfolio
resCapital = CAPITAL - sum(portfolio['value'])
pfRecord = iniPortfolio
pfValues = pd.DataFrame([{'date' : datetime.datetime(2013, 11, 1), 'value' : CAPITAL},])

StockPricesTest = pd.merge(StockPrices[StockPrices.date >= START], avaTickers, on = 'ticker')
StockReturnsTest = pd.merge(StockReturns[StockReturns.endDate >= START], avaTickers, on = 'ticker')
testDates = StockReturnsTest.endDate.map(lambda t: t.date()).unique()

##testDates = testDates[:5]
for date in testDates:
    print date
    tmpPrice = StockPricesTest[StockPricesTest.date == date][['ticker', 'adj_close']]
    tmpReturn = StockReturnsTest[StockReturnsTest.endDate == date]
    tmpData = pd.merge(portfolio.drop(['adj_close', 'date'],1), tmpReturn, on = 'ticker')
    tmpData = pd.merge(tmpData, tmpPrice, on = 'ticker')
    tmpData['value'] = tmpData['value'] * tmpData['return'].map(math.exp)
    tmpValue = sum(tmpData['value']) + resCapital
    tmpData['weight'] = tmpData['value'] / tmpValue
    tmpW = tmpData['weight'].as_matrix()
    tmpSens = np.dot(tmpW.T, mCoef)
    tmpBeta = np.dot(tmpW.T, mCqaBeta)
    endDate = tmpReturn['endDate'].iget(0)
    tmpData['date'] = endDate
    tmpData = tmpData[['ticker', 'date', 'weight', 'adj_close', 'nShare', 'value']]
    portfolio = tmpData
    pfRecord = pfRecord.append(tmpData)
    pfValues = pfValues.append(pd.DataFrame([{'date' : endDate,
                                              'value' : tmpValue,
                                              'beta_CQA' : tmpBeta,
                                              'beta_Mkt-RF' : tmpSens[0],
                                              'beta_SMB' : tmpSens[1],
                                              'beta_HML' : tmpSens[2],
                                              'beta_UMD' : tmpSens[3]},]))

pfValues = pfValues.merge(SP500Prices[['adj_close', 'date']], on = 'date')
pfValues = pfValues.rename(columns = {'adj_close':'SP500'})

def summarize(pfValues):
    plt.figure(1)
    plt.subplot(321)
    plt.ylabel('Account Value vs S&P500')
    plt.grid()
    plt.plot(pfValues.date, pfValues.value, 'r-', pfValues.date, pfValues.SP500 * CAPITAL/pfValues.SP500[0], 'b-')
    plt.axhline(CAPITAL)
    plt.subplot(322)
    plt.ylabel('beta_CQA')
    plt.grid()
    plt.plot(pfValues.date, pfValues.beta_CQA)
    plt.axhline(BETA_BOUND)
    plt.axhline(-BETA_BOUND)
    plt.subplot(323)
    plt.ylabel('beta_Mkt-RF')
    plt.grid()
    plt.plot(pfValues.date, pfValues['beta_Mkt-RF'])
    plt.subplot(324)
    plt.ylabel('beta_SMB')
    plt.grid()
    plt.plot(pfValues.date, pfValues.beta_SMB)
    plt.subplot(325)
    plt.ylabel('beta_HML')
    plt.grid()
    plt.plot(pfValues.date, pfValues.beta_HML)
    plt.subplot(326)
    plt.ylabel('beta_UMD')
    plt.grid()
    plt.plot(pfValues.date, pfValues.beta_UMD)
    plt.show()
    
summarize(pfValues)

## Now let's look at the individual stock value time series
def viewStocks(pfRecord, avaTickers):
    for ticker in avaTickers.ticker:
        print ticker
        tmpData = pfRecord[pfRecord.ticker == ticker]
        plt.plot(tmpData.date, tmpData.value)
    plt.show()

viewStocks(pfRecord, avaTickers)

findBugStock = pfRecord[pfRecord.value < -1000000]

