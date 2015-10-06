import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import ReadData
import functions
# import RiskModelTest

START_TRAIN = datetime.datetime(2011, 11, 1)
END_TRAIN = datetime.datetime(2014, 11, 6)
BUILD_DATE = datetime.datetime(2014,11, 7)
BETA_BOUND = 0.4
WEIGHT_BOUND = 0.02
CAPITAL = 1e7

DataFolder = 'D:\Dropbox\CQA 2014\Data'
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
SP500Returns = ReadData.GetSP500Return(SP500Prices)
tickers = ReadData.ReadTickers(DataFolder)

decay = 0.003 # Cheating...# Half-Value Period is 100 trading days
(Coef, Res) = functions.wReg(StockReturns, CarhartDaily, tickers, decay, START_TRAIN, END_TRAIN)
Coef.to_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay), index=False)
Res.to_csv(r'%s/Res_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay), index=False)
# Coef = pd.read_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))
# Res = pd.read_csv(r'%s/Res_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))


betas = Coef.merge(tickers).rename(columns={'Mkt-RF': 'beta'})[['ticker', 'beta']].reset_index(drop=True)
sortCoef = betas.sort('beta').reset_index(drop=True)
N = sortCoef.shape[0] / 10
longTickers = sortCoef.head(4*N).tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
shortTickers = sortCoef.tail(4*N)[['ticker']].sort('ticker').reset_index(drop=True)
pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]), axis=0).sort('ticker').reset_index(drop=True)
# def getCovMatrix(tickers):

CarhartSample = CarhartDaily[(CarhartDaily['date'] >= START_TRAIN) &
                            (CarhartDaily['date'] <= END_TRAIN)][['Mkt-RF', 'SMB', 'HML', 'UMD']].reset_index(drop=True)
ResSample = Res[pfTickers.ticker]
FactorCov = np.cov(CarhartSample.as_matrix().transpose())
ResCov = np.cov(ResSample.as_matrix().transpose())
sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers).sort('ticker').drop('ticker', axis=1).as_matrix()

totalCov = np.dot(np.dot(sens, FactorCov), sens.transpose()) + ResCov

wTickers = functions.pfOptimizer(longTickers, shortTickers, Coef, StockBeta, BETA_BOUND, WEIGHT_BOUND)
wTickers2 = wTickers[np.abs(wTickers['weight']) > 0.0005]
wTickers2['value'] = wTickers2['weight'] * CAPITAL
wTickers2.to_csv(r'%s/Portfolio_20141110_3.csv' % DataFolder, index=False)

iniPrices = pd.merge(StockPrices[StockPrices.date == BUILD_DATE][['ticker', 'date', 'adj_close']],
                     wTickers[['ticker']], on='ticker').reset_index(drop=True)
wCoef = Coef.merge(wTickers2).merge(StockBeta)

iniCoef = pd.DataFrame(np.dot(wCoef[['weight']].as_matrix().transpose(),
                              wCoef[['cqaBeta', 'Mkt-RF', 'SMB', 'HML', 'UMD']].as_matrix()))
iniCoef.columns = ['cqaBeta', 'beta', 'SMB', 'HML', 'UMD']

# Stocks with betas
# tmpData = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers)
# groupBeta = pd.DataFrame()
groupBeta = []
for n in range(10):
    sample = sortCoef.head(N * (n+1)).tail(N)
    groupBeta.append(np.mean(sample['beta']))
groupBeta