# This module calculates the covariance matrices

import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime
import ReadData
import math
import matplotlib.pyplot as plt

DataFolder = 'E:\Dropbox\CQA 2014\Data'
# BUILD_DATE = datetime.datetime(2013, 11, 1)
START_TRAIN = datetime.datetime(2010, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
START = datetime.datetime(2013, 11, 1)
END = datetime.datetime(2014, 11, 5)
BETA_BOUND = 0.45
WEIGHT_BOUND = 0.05
CAPITAL = 1e7

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
tickers = ReadData.ReadTickers(DataFolder)

decay  = 0.003
Coef = pd.read_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))
Res = pd.read_csv(r'%s/Res_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))

betas = Coef.merge(tickers).rename(columns={'Mkt-RF': 'beta'})[['ticker', 'beta']].reset_index(drop=True)
sortCoef = betas.sort('beta').reset_index(drop=True)
N = sortCoef.shape[0] / 10
longTickers = sortCoef.head(4*N).tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
shortTickers = sortCoef.tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]), axis=0).sort('ticker').reset_index(drop=True)
# def getCovMatrix(tickers):

CarhartSample = CarhartDaily[(CarhartDaily['date'] >= START_TRAIN) &
                            (CarhartDaily['date'] <= END_TRAIN)][['Mkt-RF', 'SMB', 'HML', 'UMD']].reset_index(drop=True)
ResSample = Res[pfTickers.ticker]
FactorCov = np.cov(CarhartSample.as_matrix().transpose())
ResCov = np.diag(np.diag(np.cov(ResSample.as_matrix().transpose())))
sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers).sort('ticker').drop('ticker', axis=1).as_matrix()

totalCov = np.dot(np.dot(sens, FactorCov), sens.transpose()) + ResCov