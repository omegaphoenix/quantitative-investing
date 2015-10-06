__author__ = 'Administrator'
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import ReadData
import matplotlib
import datetime
import pandas as pd
import math
import matplotlib.ticker as mtick
import functions
from matplotlib.ticker import MaxNLocator
import statsmodels.api as sm
from pylab import *
# this file produces plots for the pitch video

DataFolder = 'D:\Dropbox\CQA 2014\Data'
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
SP500Returns = ReadData.GetSP500Return(SP500Prices)
print "Finished reading data..."
cit_color = '#FF6E1E'

START_TRAIN = datetime.datetime(2010, 11, 1)
END_TRAIN = datetime.datetime(2013, 10, 31)
START = datetime.datetime(2013, 11, 1)
END = datetime.datetime(2014, 10, 31)

Fama_French_table1_return = [1.34, 1.29, 1.36, 1.31, 1.33, 1.28, 1.24, 1.21, 1.25, 1.14]
BAB_table3_return = [0.91, 0.98, 1.00, 1.03, 1.05, 1.10, 1.05, 1.08, 1.06, 0.97]
MAX_table1_return = [0.69, 0.78, 0.78, 0.77, 0.81, 0.73, 0.71, 0.65, 0.51, 0.35]


def return_bar_plot(returns, axis):
    matplotlib.rc('xtick', labelsize=36)
    matplotlib.rc('ytick', labelsize=36)
    index = np.arange(10) + 1
    bar_width = 0.4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(index, returns, color=cit_color)
    fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    # xTickMarks = ['Group1'] + ['Group' + str(i) for i in range(1, 11)]
    xTickMarks = ['Group1\n(Low)'] + [str(i) for i in range(2, 10)] + ['Group10\n(High)']
    # plt.xlabel('Beta Groups')
    # plt.ylabel('Monthly Return')
    plt.axis(axis)
    # xTickNames = ax.set_xticklabels = xTickMarks
    # plt.setp(xTickNames)
    # plt.xticks(index, xTickMarks, rotation=45)
    # plt.xticks(index+bar_width, xTickMarks, rotation=45)
    plt.xticks(index+bar_width, xTickMarks)
    plt.title('Monthly Returns of Beta Groups (low to high)', fontsize=36)
    red_patch = mpatches.Patch(color=cit_color, label='Monthly Return (percentage)')
    # plt.legend(handles=[red_patch], prop={'size': 36})
    plt.gcf().subplots_adjust(bottom=0.12)
    plt.grid()
    plt.show()

def plot_sp500():
    SP500PricesSample = SP500Prices[SP500Prices['date'] >= datetime.datetime(1970, 1, 1)]
    plt.figure()
    matplotlib.rc('xtick', labelsize=36)
    matplotlib.rc('ytick', labelsize=36)
    plt.plot(SP500PricesSample['date'], SP500PricesSample['adj_close'],
             label='S&P 500', linewidth=4, color=cit_color)
    plt.legend(loc=2, prop={'size': 36})
    plt.grid()
    plt.show()

def plot_return_vs_beta():
    Coef = ReadData.GetCoef(START_TRAIN, END_TRAIN, StockReturns, CarhartDaily, SP500Returns, DataFolder)
    avaTickers = pd.merge(Coef[['ticker']], StockBeta[ np.isnan(StockBeta.cqaBeta) == False][['ticker']])
    avaTickers = pd.merge(avaTickers, StockPrices[StockPrices.date == START_TRAIN][['ticker']], on = 'ticker')
    avaTickers = pd.merge(avaTickers, StockPrices[StockPrices.date == END_TRAIN][['ticker']], on = 'ticker')
    avaTickers = pd.merge(avaTickers, StockPrices[StockPrices.date == START][['ticker']], on = 'ticker')
    avaTickers = pd.merge(avaTickers, StockPrices[StockPrices.date == END][['ticker']], on = 'ticker')
    Coef = Coef.merge(avaTickers, on = 'ticker')

    stock_returns_sample = StockReturns[(StockReturns['endDate'] >= START) &
                                        (StockReturns['endDate'] <= END)]
    stock_annual_returns = stock_returns_sample.groupby('ticker').sum().reset_index()
    stock_annual_returns['return'] = stock_annual_returns['return'].map(math.exp)-1

    beta_return_comp = Coef[['ticker', 'Mkt-RF']].merge(stock_annual_returns)
    # plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    # plt.scatter(beta_return_comp['Mkt-RF'], beta_return_comp[ 'return'], marker='x')
    ax.scatter(beta_return_comp['Mkt-RF'], beta_return_comp['return']*100,
               marker='o', s=40, color='#FF6E1E')
    matplotlib.rc('xtick', labelsize=36)
    matplotlib.rc('ytick', labelsize=36)
    plt.xlabel('Stock Beta', fontsize=36)
    plt.ylabel('Annual Return (in percentage)', fontsize=36)
    plt.grid()
    plt.title('Return vs Beta (Nov 2013-Oct 2014)', fontsize=36)
    plt.gcf().subplots_adjust(left=0.15, bottom=0.14)
    plt.show()

def plot_performance():
    Coef = pd.read_csv(r'%s\Coef_2011-11-01_2014-11-06_0.0030.csv' % DataFolder)
    CAPITAL = 1e7
    start_date = datetime.datetime(2014, 11, 10)
    end_date = datetime.datetime(2014, 12, 16)
    account_value = pd.read_csv(r'%s/AccountValues.csv' % DataFolder, parse_dates=[0])
    account_value = account_value[(account_value['Date'] >= start_date) &
                                  (account_value['Date'] <= end_date)].ix[:, 0:2]
    account_value.columns = ['date', 'value']
    account_value['value'] = account_value['value'].map(lambda s: float(s.replace(',','')[1:]))

    sp500_value = SP500Prices[(SP500Prices['date'] >= start_date) &
                                  (SP500Prices['date'] <= end_date)]
    ini_sp500 = sp500_value['adj_close'].iloc[0]
    sp500_value['sp500value'] = sp500_value['adj_close'] / ini_sp500 * CAPITAL
    account_value = account_value.merge(sp500_value, on='date')
    account_value['return'] = (account_value['value']-CAPITAL)/CAPITAL
    account_value['sp500return'] = (account_value['sp500value']-CAPITAL)/CAPITAL

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    matplotlib.rc('xtick', labelsize=36)
    matplotlib.rc('ytick', labelsize=36)
    plt.plot(account_value['date'], account_value['return']*100, color=cit_color, linewidth=6,
             label='Portfolio Value')
    plt.plot(account_value['date'], account_value['sp500return']*100, color='b', linewidth=6,
             label='S&P 500')
    plt.legend(loc=3, prop={'size': 36})
    fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    plt.xticks(rotation=15)
    ax.yaxis.set_major_formatter(yticks)
    # ax.xaxis.set_major_locator(MaxNLocator(4))
    # xlabels = account_value['date'].map(lambda t: str(t.strftime("%b"))+ " "  +str(t.day))
    # xlabels = account_value['date'].map(lambda t: str(t.date()))
    # ax.set_xticklabels(xlabels)
    plt.gcf().subplots_adjust(bottom=0.13)
    plt.grid()
    plt.show()

def sp500_sharpe_calculator():
    sp500_returns = SP500Returns['SP500Return']
    sp500_mean_return = sp500_returns.mean() * 100 * 252 # arithmetic average
    sp500_geo_mean_return = (sp500_returns.map(lambda t: 1+t).prod())**(1/65.0)
    sp500_vol = sp500_returns.std() * 100 * math.sqrt(252)
    sp500_sharpe = sp500_mean_return/sp500_vol
    sp500_sharpe

def risk_return_expectation():
    back_date = datetime.datetime(2010, 11, 1)
    start_date = datetime.datetime(2013, 11, 1)
    end_date = datetime.datetime(2014, 10, 31)
    tickers = ReadData.ReadTickers(DataFolder)
    BETA_BOUND = 0.4
    WEIGHT_BOUND = 0.02
    decay = 0.003

    (Coef, Res) = functions.wReg(StockReturns, CarhartDaily, tickers, decay, back_date, start_date)
    betas = Coef.merge(tickers).rename(columns={'Mkt-RF': 'beta'})[['ticker', 'beta']].reset_index(drop=True)
    sortCoef = betas.sort('beta').reset_index(drop=True)
    N = sortCoef.shape[0] / 10
    longTickers = sortCoef.head(4*N).tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
    shortTickers = sortCoef.tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
    wTickers = functions.pfOptimizer(longTickers, shortTickers,
                                                  Coef, StockBeta,
                                                  BETA_BOUND, WEIGHT_BOUND)
    wTickers = wTickers[(wTickers['weight'] >= 0.001) | (wTickers['weight'] <= -0.001)]
    total_value = 1e7
    sp500_value = 1e7

    # Initial Exposures
    Exposure = wTickers.merge(Coef)
    ini_beta = (Exposure['weight']*Exposure['Mkt-RF']).sum()
    ini_smb = (Exposure['weight']*Exposure['SMB']).sum()
    ini_hml = (Exposure['weight']*Exposure['HML']).sum()
    ini_umd = (Exposure['weight']*Exposure['UMD']).sum()
    print 'ini_beta = %.2f' % ini_beta
    print 'ini_smb = %.2f' % ini_smb
    print 'ini_hml = %.2f' % ini_hml
    print 'ini_umd = %.2f' % ini_umd

    # test portfolio
    pfValues = functions.testPortfolio2(StockPrices, SP500Prices, StockBeta, Coef, wTickers,
                                        total_value, sp500_value, start_date, end_date).rename(columns={'SP500value': 'sp500value'})
    pfValues['return'] = (pfValues['value']-total_value)/total_value
    pfValues['sp500return'] = (pfValues['sp500value']-total_value)/total_value

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    matplotlib.rc('xtick', labelsize=36)
    matplotlib.rc('ytick', labelsize=36)
    plt.plot(pfValues['date'], pfValues['return']*100, color=cit_color, linewidth=6,
             label='Portfolio Value')
    plt.plot(pfValues['date'], pfValues['sp500return']*100, color='b', linewidth=6,
             label='S&P 500')
    plt.legend(loc=2, prop={'size': 36})
    fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    plt.xticks(rotation=25)
    ax.yaxis.set_major_formatter(yticks)
    plt.gcf().subplots_adjust(bottom=0.18)
    plt.grid()
    plt.show()

    breakpoints = [datetime.datetime(2013, 11, 30),
                   datetime.datetime(2013, 12, 31),
                   datetime.datetime(2014, 1, 31),
                   datetime.datetime(2014, 2, 28),
                   datetime.datetime(2014, 3, 31),
                   datetime.datetime(2014, 4, 30),
                   datetime.datetime(2014, 5, 31),
                   datetime.datetime(2014, 6, 30),
                   datetime.datetime(2014, 7, 31),
                   datetime.datetime(2014, 8, 31),
                   datetime.datetime(2014, 9, 30),
                   datetime.datetime(2014, 10, 31),
                   ]

    comparison = pd.DataFrame()
    tmp_start = datetime.datetime(2013, 10, 30)
    for tmp_end in breakpoints:
        print tmp_end
        tmp_pfValues = pfValues[(pfValues['date'] > tmp_start) &
                                (pfValues['date'] <= tmp_end)]
        # tmp_return = tmp_pfValues['return'].sum()
        # tmp_sp500_return = tmp_pfValues['sp500return'].sum()
        start_value = tmp_pfValues['value'].iloc[0]
        end_value = tmp_pfValues['value'].iloc[-1]
        tmp_return = ((end_value - start_value) / start_value) * 100
        start_sp500 = tmp_pfValues['sp500value'].iloc[0]
        end_sp500 = tmp_pfValues['sp500value'].iloc[-1]
        tmp_sp500_return = ((end_sp500 - start_sp500) / start_value) * 100
        comparison = comparison.append(pd.DataFrame([{'return': tmp_return,
                                                     'sp500return': tmp_sp500_return
                                                     }]))
        tmp_start = tmp_end

    comparison

    pfReturns = pfValues[['date', 'value', 'sp500value']]
    pfReturns['daily_return'] = (pfReturns['value']-pfReturns['value'].shift(1))/pfReturns['value'].shift(1)
    pfReturns['daily_sp500return'] = (pfReturns['sp500value']-pfReturns['sp500value'].shift(1))/pfReturns['value'].shift(1)
    pfReturns = pfReturns[1:]

    # Annualized volatility
    vol_portfolio = pfReturns['daily_return'].std()*math.sqrt(252)
    vol_sp500 = pfReturns['daily_sp500return'].std()*math.sqrt(252)
    n_days = np.float(pfReturns.shape[0])
    ave_return = ((pfValues['value'].iloc[-1]/pfValues['value'].iloc[0])**(1/n_days)-1)*100
    ave_sp500_return = ((pfValues['sp500value'].iloc[-1]/pfValues['sp500value'].iloc[0])**(1/n_days)-1)*100
    cum_return = (pfValues['value'].iloc[-1]-pfValues['value'].iloc[0])/pfValues['value'].iloc[0]*100
    cum_sp500return = (pfValues['sp500value'].iloc[-1]-pfValues['sp500value'].iloc[0])/pfValues['sp500value'].iloc[0]*100
    print 'cum_return = %.2f' % cum_return
    print 'cum_sp500return = %.2f' % cum_sp500return
    print 'vol_portfolio = %.4f' % vol_portfolio
    print 'vol_sp500 = %.4f' % vol_sp500

    sharpe_portfolio = pfReturns['daily_return'].mean()/vol_portfolio * 252
    sharpe_sp500 = pfReturns['daily_sp500return'].mean()/vol_sp500  * 252

    down_dev_portfolio = math.sqrt(pfReturns[pfReturns['daily_return']<0]['daily_return'].map(lambda x: x**2).sum()/n_days)
    down_dev_sp500 = math.sqrt(pfReturns[pfReturns['daily_sp500return']<0]['daily_sp500return'].map(lambda x: x**2).sum()/n_days)

    print 'down_dev_portfolio = %.4f' % down_dev_portfolio
    print 'down_dev_sp500 = %.4f' % down_dev_sp500

    max_drawdown = 0
    max_drawdown_sp500 = 0

    for i in range(pfValues.shape[0]-1):
        for j in range(i+1, pfValues.shape[0]):
            tmp_drawdown = (pfValues['value'].iloc[j]-pfValues['value'].iloc[i])/pfValues['value'].iloc[i]*100
            tmp_drawdown_sp500 = (pfValues['sp500value'].iloc[j]-pfValues['sp500value'].iloc[i])/pfValues['sp500value'].iloc[i]*100
            max_drawdown = min(max_drawdown, tmp_drawdown)
            max_drawdown_sp500 = min(max_drawdown_sp500, tmp_drawdown_sp500)

    print 'max_drawdown = %.2f' % max_drawdown
    print 'max_drawdown_sp500 = %.2f' % max_drawdown_sp500

    pfReturns = pfReturns.merge(CarhartDaily, on='date')
    y = pfReturns['daily_return'] * 100 - pfReturns['RF']
    X = sm.add_constant(pfReturns[['Mkt-RF', 'SMB', 'HML', 'UMD']])
    model = sm.OLS(y, X).fit()
    print model.params


def pie_chart():
    PnL_beta = 86791.68
    PnL_alpha = 179697.16
    data = [PnL_alpha, PnL_beta]
    colors = [cit_color, 'blue']
    plt.pie(data, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0.02,0))
    font_size = 24
    rcParams['font.size'] = font_size
    rcParams['axes.titlesize'] = font_size
    rcParams['xtick.labelsize'] = font_size
    rcParams['legend.fontsize'] = font_size
    labels = ['alpha', 'beta']
    plt.legend(labels, loc=1)
    plt.show()



# return_bar_plot(Fama_French_table1_return, axis=[1, 11, 1.12, 1.38])
# return_bar_plot(BAB_table3_return, axis=[1, 11, 0.9, 1.12])
# return_bar_plot(MAX_table1_return, axis=[1, 11, 0.33, 0.88])
# plot_sp500()
plot_return_vs_beta()
# plot_performance()
# risk_return_expectation()
pie_chart()