# This file finds better beta estimate
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import ReadData
import functions

DataFolder = 'E:\Dropbox\CQA 2014\Data'
START_TRAIN = datetime.datetime(2010, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
START = datetime.datetime(2013, 11, 1)
END = datetime.datetime(2014, 11, 5)

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
SP500Returns = ReadData.GetSP500Return(SP500Prices)
tickers = ReadData.ReadTickers(DataFolder)

# StockReturnsTrain = StockReturns[(StockReturns['endDate'] >= START_TRAIN) &
#                                 (StockReturns['endDate'] <= END_TRAIN)].merge(tickers)
# StockReturnsTest = StockReturns[StockReturns['endDate'] >= START].merge(tickers)


decayList = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]
# decay = functions.GetDecay(StockReturns, CarhartDaily, decayList, START_TRAIN, END_TRAIN, START, END, tickers)
decay = 0.003 # Cheating...
(Coef, Res) = functions.wReg(StockReturns, CarhartDaily, tickers, decay, START_TRAIN, END_TRAIN)

Coef.to_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay), index=False)
Res.to_csv(r'%s/Res_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay), index=False)
# comp = StockBeta.merge(CoefTrain[['ticker', 'betaTrain']]).merge(CoefTest[['ticker', 'betaTest']])
# plt.figure(1)
# plt.subplot(221)
# plt.xlabel('cqaBeta')
# plt.ylabel('betaTrain')
# plt.scatter(comp['cqaBeta'], comp['betaTrain'])
# plt.plot(range(4), range(4), 'r-')
# plt.subplot(222)
# plt.xlabel('cqaBeta')
# plt.ylabel('betaTest')
# plt.scatter(comp['cqaBeta'], comp['betaTest'])
# plt.plot(range(4), range(4), 'r-')
# plt.subplot(223)
# plt.xlabel('betaTrain')
# plt.ylabel('betaTest')
# plt.scatter(comp['betaTrain'], comp['betaTest'])
# plt.plot(range(4), range(4), 'r-')
# plt.show()
#
# diff = comp['betaTrain'] - comp['betaTest']
# diff.describe()
# error = np.linalg.norm(diff) / math.sqrt(diff.shape[0])
# error