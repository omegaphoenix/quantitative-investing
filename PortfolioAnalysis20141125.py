import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import ReadData
import functions
# import RiskModel

START_TRAIN = datetime.datetime(2011, 11, 1)
END_TRAIN = datetime.datetime(2014, 11, 6)
BUILD_DATE = datetime.datetime(2014,11, 7)
BETA_BOUND = 0.4
WEIGHT_BOUND = 0.02
CAPITAL = 1e7
decay = 0.003

DataFolder = 'D:\Dropbox\CQA 2014\Data'
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
SP500Returns = ReadData.GetSP500Return(SP500Prices)
tickers = ReadData.ReadTickers(DataFolder)

Coef = pd.read_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay),
                              parse_dates = [0])
wTickers2 = pd.read_csv(r'%s/Portfolio_20141110.csv' % DataFolder)

wCoef = Coef.merge(wTickers2).merge(StockBeta)
# wTickers = wTickers.merge(Coef)

iniCoef = pd.DataFrame(np.dot(wCoef[['weight']].as_matrix().transpose(),
                              wCoef[['cqaBeta', 'Mkt-RF', 'SMB', 'HML', 'UMD']].as_matrix()))
iniCoef.columns = ['cqaBeta', 'beta', 'SMB', 'HML', 'UMD']

