# This file tests the actual performance of our portfolio
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import ReadData
import functions

DataFolder = 'D:\Dropbox\CQA 2014\Data'
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
Coef = pd.read_csv(r'%s\Coef_2011-11-01_2014-11-06_0.0030.csv' % DataFolder)
CAPITAL = 1e7
build_date = datetime.datetime(2014, 11, 10)
end_date = datetime.datetime(2014, 11, 30)
# StockReturns = ReadData.ReadReturn(DataFolder)
# SP500Returns = ReadData.GetSP500Return(SP500Prices)

StockPricesSample = StockPrices[(StockPrices['date'] >= build_date) &
                                (StockPrices['date'] <= end_date)]

open_positions = pd.read_excel(r'%s\OpenPositions_12-10-2014.xlsx' % DataFolder)
portfolio_20141110 = open_positions[['Symbol', 'QTY', 'Price Paid']]\
    .rename(columns={'Symbol': 'ticker', 'QTY': 'nShare', 'Price Paid': 'price'}).replace('BFB', 'BF.B')
portfolio_20141110['value'] = portfolio_20141110['nShare'] * portfolio_20141110['price']

# # test if there's any ticker missing
# all_10y_tickers = pd.DataFrame(StockPricesSample['ticker'].unique())
# all_tickers = pd.DataFrame(StockPrices['ticker'].unique())
# all_10y_tickers.columns = ['ticker']
# all_tickers.columns = ['ticker']
# pf_tickers = pd.DataFrame(portfolio_20141110['ticker'].unique())
# pf_tickers.columns = ['ticker']
# NovPrices = pd.read_csv(r'%s\PortfolioPerformance.csv' % DataFolder, parse_dates=[1]).rename(columns={'symbol': 'ticker'})
# nov_tickers = pd.DataFrame(NovPrices['ticker'].unique())
# nov_tickers.columns = ['ticker']
#
# mismatch_tickers_1 = set(nov_tickers['ticker']).difference(set(all_10y_tickers['ticker']))
# print mismatch_tickers_1
# mismatch_tickers_2 = set(all_tickers['ticker']).difference(set(all_10y_tickers['ticker']))
# print mismatch_tickers_2
#
# test_tickers = all_10y_tickers.merge(pf_tickers)
# # test_tickers = all_tickers.merge(pf_tickers)
# missing_tickers = set(pf_tickers['ticker']).difference(set(test_tickers['ticker']))
# print missing_tickers
# test_stock_price = StockPrices[StockPrices['ticker'] == 'AYI']
# print test_stock_price
# assert test_tickers.shape == pf_tickers.shape
#
# test_date = StockPricesSample['date'].map(lambda t: t.date()).unique()[0]
# test_stock_prices = StockPricesSample[StockPricesSample['date'] == test_date].merge(pf_tickers)
#
# test_stock_price = NovPrices[NovPrices['ticker'] == 'AYI']
# print test_stock_price
#
# test_nov_tickers = nov_tickers.merge(pf_tickers)
# missing_tickers = set(pf_tickers['ticker']).difference(set(test_nov_tickers['ticker']))
# assert test_nov_tickers.shape == pf_tickers.shape

ini_net_value = portfolio_20141110['value'].sum()
ini_long = portfolio_20141110['value'].map(lambda x: max(x,0)).sum()
ini_short = portfolio_20141110['value'].map(lambda x: max(-x,0)).sum()
trading_cost = (ini_long + ini_short) * 0.15/100
short_proceeds = 9797626.42
cash = 0
loan = 7156
# ini_value = CAPITAL + ini_net_value - trading_cost
ini_value = cash + short_proceeds + ini_long - ini_short - loan

# test the portfolio value formula, and it's defined to be
# pf_value = cash + short_proceeds + long_position - short_position - loan
# pf_value = cash + short_proceeds + 9875373.03 - 9397954.99 - loan
# assert (ini_long - ini_short - ini_value) < 1e-4

pfValues = functions.testPortfolio3(StockPricesSample, SP500Prices, StockBeta, Coef,
                                    portfolio_20141110, short_proceeds-loan,
                                    CAPITAL, build_date, end_date)

plt.plot(pfValues['date'], pfValues['value'], 'b', linewidth=2)
plt.plot(pfValues['date'], pfValues['SP500value'], 'r', linewidth=2)
plt.grid()
plt.show()
