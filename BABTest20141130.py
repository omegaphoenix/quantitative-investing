# This files tests our new strategy: long mid-beta stocks, short low/high-beta stocks, while
# maximizing exposure to SMB/HML/UMD.

import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import ReadData
import functions
from cvxopt import matrix, solvers, spmatrix
# from scipy.optimize import minimize

DataFolder = 'D:\Dropbox\CQA 2014\Data'
PlotFolder = 'D:\Dropbox\CQA 2014\Plots'
CAPITAL = 1e7
START_TRAIN = datetime.datetime(2004, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
BACK = datetime.datetime(2004, 11, 1)
# BACK = datetime.datetime(2010, 11, 1)
START = datetime.datetime(2007, 11, 1)
# START = datetime.datetime(2013, 11, 1)
END = datetime.datetime(2014, 12, 5)
BETA_BOUND = 0.4
WEIGHT_BOUND = 0.02
decay = 0.003

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
tickers = ReadData.ReadTickers(DataFolder)

# to speed up, we first look at one year's performance of each strategy
StockPrices = StockPrices.merge(tickers, on='ticker')
StockPrices = StockPrices[(StockPrices['date'] >= BACK) & (StockPrices['date'] <= END)].reset_index(drop=True)
StockReturns = StockReturns.merge(tickers, on='ticker')
StockReturns = StockReturns[(StockReturns['endDate'] >= BACK) & (StockReturns['endDate'] <= END)].reset_index(drop=True)

test_dates = StockPrices['date'].astype(datetime.datetime).unique()

rebalance_gap = 40
# start_index = 2280
start_index = 750
n_periods = (test_dates.shape[0] - start_index - rebalance_gap) / rebalance_gap
pfValues = pd.DataFrame()

# n_periods = 1
total_value = CAPITAL
sp500_value = CAPITAL
value_error_count = 0

for k in range(n_periods):
    # k = 13
    back_date = test_dates[start_index - 750 + k * rebalance_gap]
    start_date = test_dates[start_index + k * rebalance_gap]
    end_date = test_dates[start_index + (k+1) * rebalance_gap]
    print 'end_date = %s' % end_date
    (Coef, Res) = functions.wReg(StockReturns, CarhartDaily, tickers, decay, back_date, start_date)
    # find long/short universe
    betas = Coef.merge(tickers).rename(columns={'Mkt-RF': 'beta'})[['ticker', 'beta']].reset_index(drop=True)
    sortCoef = betas.sort('beta').reset_index(drop=True)
    N = sortCoef.shape[0] / 10
    longTickers = sortCoef.head(4*N).tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
    shortTickers = sortCoef.tail(3*N)[['ticker']].append(sortCoef.head(N)[['ticker']]).sort('ticker').reset_index(drop=True)
    shortTickersHigh = sortCoef.tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
    shortTickersLow = sortCoef.head(N)[['ticker']].sort('ticker').reset_index(drop=True)
    # optimizing portfolio
    try:
        (opt_comp, wTickers) = functions.pfOptimizer7(longTickers, shortTickersHigh, shortTickersLow,
                                                      Coef, Res, CarhartDaily, StockBeta,
                                                      BETA_BOUND, WEIGHT_BOUND, back_date, start_date)
    except ValueError:
        value_error_count += 1
        pass
    wTickers = wTickers[(wTickers['weight'] >= 0.001) | (wTickers['weight'] <= -0.001)]
    # test portfolio
    pfValues = pfValues.append(
        functions.testPortfolio2(StockPrices, SP500Prices, StockBeta, Coef, wTickers, total_value, sp500_value, start_date, end_date))
    total_value = pfValues['value'].iloc[-1]
    sp500_value = pfValues['SP500value'].iloc[-1]
pfValues.reset_index(drop=True)
# SP500Sample = SP500Prices[(SP500Prices.date >= start_date) & (SP500Prices.date <= end_date)].reset_index(drop=True)
# ini_sp500 = SP500Prices[SP500Prices.date < start_date].tail(1)['adj_close'].iloc[0]


# analyze portfolio performance in each month
def pfValuesPlot(pfValues, BETA_BOUND):
    line_width = 1.0
    plt.subplot(221)
    plt.plot(pfValues['date'], pfValues['value'], label='Portfolio Value', color='r', linewidth=line_width)
    plt.plot(pfValues['date'], pfValues['SP500value'], label='S&P500', color='b', linewidth=line_width)
    plt.title('Portfolio Value vs S&P500')
    plt.legend(bbox_to_anchor=(0.2, 1))
    plt.grid()
    plt.subplot(222)
    plt.plot(pfValues['date'], pfValues['beta_CQA'], label='beta_CQA', color='b', linewidth=line_width)
    plt.plot(pfValues['date'], pfValues['beta_Mkt-RF'], label='beta', color='r', linewidth=line_width)
    plt.title('beta_CQA and beta')
    plt.axhline(BETA_BOUND)
    plt.axhline(-BETA_BOUND)
    plt.legend(bbox_to_anchor=(0.2, 1))
    plt.grid()
    plt.subplot(223)
    plt.plot(pfValues['date'], pfValues['beta_SMB'], label='beta_SMB', color='b', linewidth=line_width)
    plt.plot(pfValues['date'], pfValues['beta_HML'], label='beta_HML', color='r', linewidth=line_width)
    plt.plot(pfValues['date'], pfValues['beta_UMD'], label='beta_UMD', color='g', linewidth=line_width)
    plt.legend(bbox_to_anchor=(0.2, 1))
    plt.title('exposure to other factors')
    plt.grid()
    plt.show()
    # plt.savefig(r'%s/StrategyTest.png' % PlotFolder, dpi=100)

def pfValuesAnalysis(pfValues):
    test_dates = pfValues['date'].astype(datetime.datetime).unique()
    years = pfValues['date'].astype(datetime.datetime).map(lambda t: t.year).unique()
    anal_res = pd.DataFrame()
    for year in years:
        for month in range(1,13):
            start_date = datetime.datetime(year, month, 1)
            if start_date > test_dates[-10]:
                return anal_res
            if month < 12:
                end_date = datetime.datetime(year, month+1, 1)
            else:
                end_date = datetime.datetime(year+1, 1, 1)
            if end_date < test_dates[10]:
                continue
            sample = pfValues[(pfValues['date'] >= start_date) & (pfValues['date'] <= end_date)]
            # print sample
            total_return = np.log(sample['value'].iloc[-1] / sample['value'].iloc[0])
            sp500_return = np.log(sample['SP500'].iloc[-1] / sample['SP500'].iloc[0])
            sample_return = pd.concat([pd.DataFrame(np.log(sample['value']/sample['value'].shift(1))[1:]),
                                       pd.DataFrame(np.log(sample['SP500']/sample['SP500'].shift(1))[1:])], axis=1)
            sample_return.columns = ['return', 'SP500return']
            sample_return['diff_return'] = sample_return['return'] - sample_return['SP500return']
            corr_matrix = sample_return[['return', 'SP500return']].corr()
            anal_res = anal_res.append(pd.DataFrame([{'month': datetime.date(year, month, 1),
                                                      'return': total_return,
                                                      'std': sample_return['return'].std(),
                                                      'max_day_loss': sample_return['return'].min(),
                                                      'max_day_gain': sample_return['return'].max(),
                                                      'return_over_sp500': total_return - sp500_return,
                                                      'std_over_sp500': sample_return['diff_return'].std(),
                                                      'corr': corr_matrix['return'].iloc[1],
                                                      'sharpe_annual': sample_return['return'].mean()/sample_return['return'].std() * math.sqrt(252)
                                                      }]))
            anal_res = anal_res[['month', 'return', 'std', 'sharpe_annual',
                                 'max_day_gain', 'max_day_loss',
                                 'return_over_sp500',
                                 'std_over_sp500']]
    return anal_res.reset_index(drop=True)

pfValuesPlot(pfValues, BETA_BOUND)
anal_res = pfValuesAnalysis(pfValues)
anal_res.to_csv(r'%s\Portfolio_Analysis.csv' % DataFolder,
                    index=False)
print 'value error count = %d' % value_error_count