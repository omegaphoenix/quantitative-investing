# this file generates portfolio report for CQA challenge
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axis as ax
import datetime
import ReadData
import functions
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

data_folder = 'D:\Dropbox\CQA 2014\Data'
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(data_folder)


def read_account_value(SP500Prices, CarhartDaily, start_date, end_date):
    account_value = pd.read_csv(r'%s/AccountValues.csv' % data_folder, parse_dates=[0])
    CAPITAL = 1e7

    account_value = account_value[(account_value['Date'] >= start_date) &
                                  (account_value['Date'] <= end_date)].ix[:, 0:2]
    account_value.columns = ['date', 'value']
    account_value['value'] = account_value['value'].map(lambda s: float(s.replace(',', '')[1:]))

    sp500_value = SP500Prices[(SP500Prices['date'] >= start_date) &
                              (SP500Prices['date'] <= end_date)].reset_index(drop=True)
    ini_sp500 = sp500_value['adj_close'].iloc[0]
    sp500_value['sp500value'] = sp500_value['adj_close'] / ini_sp500 * CAPITAL
    account_value = account_value.merge(sp500_value, on='date')
    account_value['pf_return'] = account_value['value'] / account_value['value'].iloc[0] - 1
    account_value['sp500_return'] = account_value['sp500value'] / \
                                    account_value['sp500value'].iloc[0] - 1
    account_value['return'] = account_value['value'] / account_value['value'].shift(1) - 1
    account_value = account_value.merge(CarhartDaily)
    return account_value


start_date = datetime.datetime(2014, 11, 7)
end_date = datetime.datetime(2015, 1, 31)
CAPITAL = 1e7
cit_color = '#FF6E1E'
account_value = read_account_value(SP500Prices, CarhartDaily, start_date, end_date)
# ##########################################################
# plot portfolio value vs SP&500


def plot_vs_sp500(account_value, font_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)
    # plt.plot(account_value['date'], account_value['value'], color=cit_color, linewidth=6,
    # label='Caltech Portfolio')
    plt.plot(account_value['date'], account_value['pf_return'] * 100, color=cit_color, linewidth=6,
             label='Caltech Portfolio')
    # plt.plot(account_value['date'], account_value['sp500value'], color='b', linewidth=4,
    # label='S&P 500')
    plt.plot(account_value['date'], account_value['sp500_return'] * 100, color='b', linewidth=4,
             label='S&P 500')
    plt.legend(loc=2, prop={'size': 36})
    fmt = '%.2f%%'  # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    plt.ylabel('Return', size=font_size)
    plt.grid()
    plt.show()


# plot_vs_sp500(account_value, font_size=24)
# ##########################################################
# get 4-factor alpha


def carhart_regression(account_value):
    reg_data = account_value[['date', 'return', 'Mkt-RF', 'SMB', 'HML', 'UMD', 'RF']].ix[1:]
    reg_data['adj_return'] = reg_data['return'] * 100 - reg_data['RF']

    y = reg_data['adj_return']
    X = sm.add_constant(reg_data[['Mkt-RF', 'SMB', 'HML', 'UMD']])
    model = sm.OLS(y, X).fit()
    print model.summary()
    alpha_daily = model.params['const']
    # print 'annualized alpha = %0.2f%%' % (((1.0 + alpha_daily / 100) ** 250 - 1) * 100)
    print 'annualized alpha = %0.2f%%' % (alpha_daily * 252)


def plot_vs_ff_factors(account_value):
    tmp_data = account_value[['date', 'pf_return', 'value', 'Mkt-RF', 'SMB', 'HML', 'UMD']]
    tmp_data['mkt_value'] = CAPITAL * \
                            np.cumprod((1.0 + tmp_data['Mkt-RF'].shift(-1) / 100).shift(1))
    tmp_data['mkt_value'].iloc[0] = CAPITAL
    tmp_data['mkt_return'] = tmp_data['mkt_value'] / tmp_data['mkt_value'].iloc[0] - 1
    tmp_data['smb_value'] = CAPITAL * \
                            np.cumprod((1.0 + tmp_data['SMB'].shift(-1) / 100).shift(1))
    tmp_data['smb_value'].iloc[0] = CAPITAL
    tmp_data['smb_return'] = tmp_data['smb_value'] / tmp_data['smb_value'].iloc[0] - 1
    tmp_data['hml_value'] = CAPITAL * \
                            np.cumprod((1.0 + tmp_data['HML'].shift(-1) / 100).shift(1))
    tmp_data['hml_value'].iloc[0] = CAPITAL
    tmp_data['hml_return'] = tmp_data['hml_value'] / tmp_data['hml_value'].iloc[0] - 1
    tmp_data['umd_value'] = CAPITAL * \
                            np.cumprod((1.0 + tmp_data['UMD'].shift(-1) / 100).shift(1))
    tmp_data['umd_value'].iloc[0] = CAPITAL
    tmp_data['umd_return'] = tmp_data['umd_value'] / tmp_data['umd_value'].iloc[0] - 1
    font_size = 24
    fig = plt.figure()
    ax = fig.add_subplot(111)
    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)
    # plt.plot(tmp_data['date'], tmp_data['value'], color=cit_color, linewidth=6,
    # label='Caltech Portfolio')
    # plt.plot(tmp_data['date'], tmp_data['smb_value'], color='FireBrick', linewidth=4,
    # label='SMB Portfolio')
    # plt.plot(tmp_data['date'], tmp_data['hml_value'], color='Fuchsia', linewidth=4,
    # label='HML Portfolio')
    # plt.plot(tmp_data['date'], tmp_data['umd_value'], color='DeepSkyBlue', linewidth=4,
    # label='UMD Portfolio')
    plt.plot(tmp_data['date'], tmp_data['pf_return'] * 100, color=cit_color, linewidth=6,
             label='Caltech Portfolio')
    plt.plot(tmp_data['date'], tmp_data['smb_return'] * 100, color='FireBrick', linewidth=4,
             label='SMB Portfolio')
    plt.plot(tmp_data['date'], tmp_data['hml_return'] * 100, color='Fuchsia', linewidth=4,
             label='HML Portfolio')
    plt.plot(tmp_data['date'], tmp_data['umd_return'] * 100, color='DeepSkyBlue', linewidth=4,
             label='UMD Portfolio')
    plt.legend(loc=2, prop={'size': font_size})
    plt.ylabel('Value', size=font_size)
    fmt = '%.2f%%'  # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    plt.grid()
    plt.show()


carhart_sample = CarhartDaily[(CarhartDaily['date'] >= start_date) &
                              (CarhartDaily['date'] <= end_date)]
# carhart_regression(account_value)
# plot_vs_ff_factors(account_value)
# ########################################################
# compare major statistics


def get_max_drawdown(value_series):
    max_drawdown = 0
    for i in range(value_series.shape[0] - 1):
        for j in range(i + 1, value_series.shape[0]):
            tmp_drawdown = (value_series.iloc[j] - value_series.iloc[i]) \
                           / value_series.iloc[i] * 100
            max_drawdown = min(max_drawdown, tmp_drawdown)
    print 'max_drawdown = %.2f' % max_drawdown
    return max_drawdown


def get_major_statistics(value_series):
    returns = ((value_series / value_series.shift(1) - 1) * 100).iloc[1:]
    down_returns = returns.loc[returns < 0]
    results = pd.DataFrame([{'total_return': (value_series.iloc[-1] / value_series.iloc[0] - 1) * 100,
                             'std': returns.std() * math.sqrt(252),
                             'max_day_loss': returns.min(),
                             'max_day_gain': returns.max(),
                             'max_drawdown': get_max_drawdown(value_series),
                             'downsize_deviation': down_returns.std() * math.sqrt(252),
                             'sharpe_annual': returns.mean() / returns.std() * math.sqrt(252)
                            }])
    return results


account_stats = get_major_statistics(account_value['value'])
sp500_stats = get_major_statistics(account_value['sp500value'])
print 'Major statistics for our portfolio:'
print account_stats
print 'Major statistics for S&P 500:'
print sp500_stats
# ########################################################
# compare performance of 10 beta groups


def get_bab_pf_values(StockReturns):
    first_date = StockPrices['date'].min()
    last_date = StockPrices['date'].max()
    stock_first = StockPrices[StockPrices['date'] == first_date][['ticker']].reset_index(drop=True)
    stock_last = StockPrices[StockPrices['date'] == last_date][['ticker']].reset_index(drop=True)
    stock_universe = stock_first.merge(stock_last).merge(StockBeta)
    tickers = stock_universe[['ticker']]
    # test_dates = np.sort(StockPrices['date'].astype(datetime.datetime).unique())
    build_start_date = datetime.datetime(2011, 11, 1)
    build_end_date = datetime.datetime(2014, 11, 7)
    test_start_date = datetime.datetime(2014, 11, 10)
    test_end_date = datetime.datetime(2015, 1, 31)
    decay = 0.003

    prices_build = StockPrices[StockPrices['date'] == build_end_date][['ticker', 'adj_close']]
    (Coef, Res) = functions.wReg(StockReturns, CarhartDaily, tickers, decay,
                                 build_start_date, build_end_date)
    betas = Coef.merge(tickers).rename(columns={'Mkt-RF': 'beta'})[['ticker', 'beta']].reset_index(drop=True)
    sortCoef = betas.sort('beta').reset_index(drop=True)
    N = sortCoef.shape[0] / 10

    font_size = 20
    # color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightblue', 'gold', 'pink']
    color_cycle = []
    for n in range(10):
        color_cycle.append((n * 0.1, 1.0 - n * 0.1, 0.0))
    pfValuesAll = pd.DataFrame()
    for n in range(10):
        tmpData = sortCoef.head(N * (n + 1)).tail(N)
        mean_beta = tmpData['beta'].mean()
        wTickers = tmpData[['ticker']].reset_index(drop=True)
        wTickers['weight'] = 1 / np.float64(N)
        portfolio = wTickers.merge(prices_build).rename(columns={'adj_close': 'price'})
        portfolio['nShare'] = (CAPITAL * portfolio['weight'] / portfolio['price']).map(functions.intPart)
        pfValues = functions.testPortfolio_simple(StockPrices, StockReturns, SP500Prices, StockBeta,
                                                  portfolio, test_start_date, test_end_date)
        pfValues['beta group'] = n
        pfValues['mean beta'] = mean_beta
        pfValues['value'] = pfValues['value'] + CAPITAL - pfValues['value'][0]
        plt.plot(pfValues['date'], pfValues['value'],
                 linewidth=2.0, color=color_cycle[n],
                 label='mean beta %0.2f' % mean_beta)
        pfValuesAll = pfValuesAll.append(pfValues)

    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)

    # plt.title('Performance of Beta Groups', size=font_size)
    plt.ylabel('Account Value', size=font_size)
    plt.xlabel('Date', size=font_size)
    plt.grid()
    plt.legend(loc=3)
    plt.show()
    return pfValuesAll


def return_bar_plot(returns, axis, font_size,
                    title_prefix='Cumulative Returns', stds=pd.Series()):
    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)
    index = np.arange(10) + 1
    bar_width = 0.4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if stds.empty:
        ax.bar(index, returns, color=cit_color)
    else:
        ax.bar(index, returns, color=cit_color, yerr=stds)
    fmt = '%.2f%%'  # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    xTickMarks = ['Group1\n(Low)'] + [str(i) for i in range(2, 10)] + ['Group10\n(High)']
    plt.axis(axis)
    plt.xticks(index + bar_width, xTickMarks)
    # plt.title('%s of Beta Groups (low to high)' % title_prefix, fontsize=font_size)
    # red_patch = mpatches.Patch(color=cit_color, label='Monthly Return (percentage)')
    # plt.legend(handles=[red_patch], prop={'size': 36})
    plt.gcf().subplots_adjust(bottom=0.12)
    plt.grid()
    plt.show()


def get_bab_groups(pfValuesAll):
    bab_groups = pd.DataFrame()
    for n in range(10):
        tmpData = pfValuesAll[pfValuesAll['beta group'] == n].sort('date', axis=0)
        tmpData['return'] = tmpData['value'] / tmpData['value'].shift(1) - 1
        reg_data = tmpData[['date', 'return']].merge(CarhartDaily).iloc[1:]
        y = reg_data['return'] * 100 - reg_data['RF']
        X = sm.add_constant(reg_data[['Mkt-RF', 'SMB', 'HML', 'UMD']])
        model = sm.OLS(y, X).fit()
        print model.summary()
        alpha_daily = model.params['const']
        # alpha_std_daily = model.bse['const']
        # print model.conf_int().iloc[1, 0]
        alpha_std_daily = (model.conf_int().iloc[0, 1] - model.conf_int().iloc[0, 0]) / 2
        # print 'annualized alpha = %0.2f%%' % (((1.0 + alpha_daily / 100) ** 250 - 1) * 100)
        print 'annualized alpha = %0.2f%%' % (alpha_daily * 252)
        bab_groups = bab_groups.append(pd.DataFrame([{'beta group': n,
                                                      'return': (tmpData['value'].iloc[-1] / tmpData['value'].iloc[
                                                          0] - 1) * 100,
                                                      # 'alpha': ((1.0 + alpha_daily / 100) ** 250 - 1) * 100,
                                                      'alpha': alpha_daily * 252,
                                                      'alpha std': alpha_std_daily * 252}]))

    bab_groups = bab_groups.reset_index(drop=True)
    return_bar_plot(bab_groups['return'], axis=[1, 11, -11, 8], font_size=24)
    return_bar_plot(bab_groups['alpha'], axis=[1, 11, -8, 35], font_size=24,
                    title_prefix='Alpha(annualized)',
                    stds=bab_groups['alpha std'])
    return bab_groups


StockReturns = ReadData.ReadReturn(data_folder)
pfValuesAll = get_bab_pf_values(StockReturns)
bab_groups = get_bab_groups(pfValuesAll)

# ########################################################
# explain performance using 10 beta groups
long_factor = pfValuesAll[pfValuesAll['beta group'] == 4][['date', 'value']]
long_factor['long return'] = long_factor['value'] / long_factor['value'].shift(1) - 1
short_factor = pfValuesAll[pfValuesAll['beta group'] == 7][['date', 'value']]
short_factor['short return'] = short_factor['value'] / short_factor['value'].shift(1) - 1
reg_data = account_value[['date', 'return']] \
    .merge(long_factor[['date', 'long return']], on='date') \
    .merge(short_factor[['date', 'short return']], on='date')
y = reg_data['return']
X = sm.add_constant(reg_data[['long return', 'short return']])
model = sm.OLS(y, X).fit_regularized()
print model.summary()
