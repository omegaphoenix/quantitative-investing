# This files tests functions in module "functions.py"
import functions
import ReadData
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DataFolder = 'E:\Dropbox\CQA 2014\Data'
CAPITAL = 1e7
# BUILD_DATE = datetime.datetime(2013, 11, 1)
START_TRAIN = datetime.datetime(2010, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
START = datetime.datetime(2013, 11, 1)
END = datetime.datetime(2014, 11, 5)
BETA_BOUND = 0.45
WEIGHT_BOUND = 0.05

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
tickers = ReadData.ReadTickers(DataFolder)
wTickers = tickers
wTickers['weight'] = 1/np.float64(wTickers.shape[0])

def testPortfolio2Test():
    pfValues = functions.testPortfolio2(StockPrices, SP500Prices, StockBeta, wTickers, CAPITAL, START, END)
    plt.subplot(121)
    plt.title('Portfolio Value vs Date')
    plt.plot(pfValues['date'], pfValues['value'],
             label='Portfolio Value', color='b')
    plt.plot(pfValues['date'], pfValues['SP500'] * (pfValues['value'].iloc[0]/pfValues['SP500'].iloc[0]),
             label='S&P500', color='r')
    plt.legend()
    plt.grid()
    plt.subplot(122)
    plt.title('beta_CQA vs Date')
    plt.plot(pfValues['date'], pfValues['beta_CQA'])
    plt.grid()
    plt.show()
    return pfValues

def get_cov_test():
    decay  = 0.003
    Coef = pd.read_csv(r'%s/Coef_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))
    Res = pd.read_csv(r'%s/Res_%s_%s_%.4f.csv' % (DataFolder, START_TRAIN.date(), END_TRAIN.date(), decay))
    total_cov = functions.get_cov(wTickers, Coef, Res, CarhartDaily, START_TRAIN, END_TRAIN)
    # plt.pcolor(total_cov)
    # plt.show()
    return total_cov

# pfValues = testPortfolio2Test()
total_cov = get_cov_test()
