# This file updates our portfolio, to maximize BAB strategy
import pandas as pd
import numpy as np
import matplotlib
import ReadData
import functions
import datetime
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers, spmatrix


# get the stock universe since a lot of tickers have disappeared
def get_stock_universe(StockPrices):
    date_list = np.sort(StockPrices['date'].unique())
    date_end = date_list[-1]
    date_start = date_list[-780]
    stock_universe_end = StockPrices[StockPrices['date'] == date_end][['ticker']].sort('ticker').reset_index(drop=True)
    stock_universe_start = StockPrices[StockPrices['date'] == date_start][['ticker']].sort('ticker').reset_index(
        drop=True)
    stock_universe = stock_universe_end.merge(stock_universe_start).sort('ticker').reset_index(drop=True)
    return stock_universe


# get stock distribution in different beta groups
def pf_beta_distribution(wTickers, sortCoef):
    N = sortCoef.shape[0] / 10
    res = pd.DataFrame()
    for n in xrange(10):
        beta_group = sortCoef.head((n + 1) * N).tail(N)[['ticker']].sort('ticker').reset_index(drop=True)
        beta_group_pf = beta_group.merge(wTickers)
        res = res.append(pd.DataFrame({'beta group': n,
                                       'count': beta_group_pf.shape[0],
                                       'weight': beta_group_pf['weight'].sum()},
                                      index=[n]))
    return res


def pf_optimizer(long_tickers, short_tickers,
                 Coef, industry_data, CarhartDaily, StockBeta,
                 beta_bound, position_bound, sector_bound, three_factor_bound, trim,
                 back_date, start_date):
    long_tickers['bWeight'] = 1  # binary weights
    short_tickers['bWeight'] = -1
    pf_tickers = pd.concat((short_tickers[['ticker']], long_tickers[['ticker']]),
                           axis=0).sort('ticker').reset_index(drop=True).merge(industry_data[['ticker', 'sector']])

    sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pf_tickers)
    m_beta = matrix(sens['Mkt-RF'])
    m_beta_smb = matrix(sens['SMB'])
    m_beta_hml = matrix(sens['HML'])
    m_beta_umd = matrix(sens['UMD'])
    m_cqa_beta = matrix(StockBeta.merge(pf_tickers)['cqaBeta'])

    longIndex = matrix(pf_tickers.merge(long_tickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pf_tickers.merge(long_tickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex, 1) > 0.5])]).trans()
    shortIndex = -matrix(pf_tickers.merge(short_tickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pf_tickers.merge(short_tickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex, 1) > 0.5])]).trans()

    sector_index = pf_tickers[['ticker', 'sector']]
    sector_list = pf_tickers['sector'].unique()
    for sector in sector_list:
        sector_index.loc[:, sector] = 0.0
        sector_index.ix[sector_index['sector'] == sector, sector] = 1.0
    mSector_index = matrix(sector_index.iloc[:, 2:].as_matrix())

    N = pf_tickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N, 1)) * position_bound)
    longBounds = matrix(np.zeros((short_tickers.shape[0], 1)))
    shortBounds = matrix(np.zeros((long_tickers.shape[0], 1)))
    total_cov = matrix(functions.get_cov(pf_tickers, Coef, Res, CarhartDaily, back_date, start_date))
    q = matrix(np.zeros((pf_tickers.shape[0], 1)))
    # three_factor_bound = 0.05
    G = matrix([[mSector_index], [-mSector_index],
                [m_beta], [-m_beta],
                [m_beta_smb], [-m_beta_smb],
                [m_beta_hml], [-m_beta_hml],
                [m_beta_umd], [-m_beta_umd],
                [m_cqa_beta], [-m_cqa_beta],
                [longIndex], [-longIndex],
                [shortIndex], [-shortIndex],
                [id], [-id],
                [-mLongIndex], [mShortIndex]]).trans()
    h = matrix([sector_bound, sector_bound, sector_bound, sector_bound,
                sector_bound, sector_bound, sector_bound, sector_bound,
                sector_bound, sector_bound, sector_bound, sector_bound,
                sector_bound, sector_bound, sector_bound, sector_bound,
                0.2, 0.4,
                three_factor_bound, three_factor_bound,
                three_factor_bound, three_factor_bound,
                three_factor_bound, three_factor_bound,
                beta_bound, beta_bound,
                1, -0.98,
                -0.98, 1,
                wBounds, wBounds,
                longBounds, shortBounds])
    sol = solvers.qp(total_cov, q, G, h)
    w_final = sol['x']
    wTickers = pf_tickers
    wTickers['weight'] = w_final
    wTickers = wTickers[np.abs(wTickers['weight']) > trim]
    wTickers_properties = wTickers.merge(Coef).merge(StockBeta)
    stats = pd.DataFrame({'beta': np.dot(wTickers_properties['weight'], wTickers_properties['Mkt-RF']),
                          'beta_smb': np.dot(wTickers_properties['weight'], wTickers_properties['SMB']),
                          'beta_hml': np.dot(wTickers_properties['weight'], wTickers_properties['HML']),
                          'beta_umd': np.dot(wTickers_properties['weight'], wTickers_properties['UMD']),
                          'beta_cqa': np.dot(wTickers_properties['weight'], wTickers_properties['cqaBeta']),
                          'long weight': wTickers[wTickers['weight'] > 0]['weight'].sum(),
                          'short weight': wTickers[wTickers['weight'] < 0]['weight'].sum()
                         }, index=[0])
    for sector in sector_list:
        stats[sector] = wTickers_properties[wTickers_properties['sector'] == sector]['weight'].sum()
    stats = stats.transpose().reset_index()
    stats.columns = ['statistics', 'value']
    return (stats, wTickers)


data_folder = 'E:/Dropbox/CQA 2014/Data'
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(data_folder)
StockReturns = ReadData.ReadReturn(data_folder)
industry_data = pd.read_csv('%s\industry_data.csv' % data_folder)
pf_0208 = pd.read_csv('%s/OpenPositions_20150208.csv' % data_folder).replace('BF.B', 'BFB')
pf_0208 = pf_0208[['Symbol', 'QTY', 'Price Paid', 'Last Price', 'Market Value']] \
    .rename(columns={'Symbol': 'ticker',
                     'QTY': 'nShare',
                     'Price Paid': 'price',
                     'Last Price': 'price_current',
                     'Market Value': 'value'}) \
    .sort('ticker').reset_index(drop=True)
dollar_to_float = lambda s: float(s.strip('$').replace(',', ''))
pf_0208['price'] = pf_0208['price'].map(dollar_to_float)
pf_0208['price_current'] = pf_0208['price_current'].map(dollar_to_float)
pf_0208['value'] = pf_0208['value'].map(dollar_to_float) * pf_0208['nShare'].map(lambda q: np.sign(q))

# single stock performance
pf_0208['return'] = np.log(pf_0208['price_current'] / pf_0208['price']) * 100 * np.sign(pf_0208['nShare'])
pf_0208['position'] = np.abs(pf_0208['value'])

# test_end_date = last_date
decay = 0.003
tickers = get_stock_universe(StockPrices)
build_start_date = datetime.datetime(2012, 2, 6)
build_end_date = datetime.datetime(2015, 2, 6)

# prices_build = StockPrices[StockPrices['date'] == build_end_date][['ticker', 'adj_close']]

(Coef, Res) = functions.wReg(StockReturns, CarhartDaily, tickers, decay,
                             build_start_date, build_end_date)
tickers_after_reg = tickers.merge(Coef[['ticker']])
tickers_industry = tickers.merge(industry_data[['ticker']])
betas = Coef.merge(tickers_industry).rename(columns={'Mkt-RF': 'beta'})[['ticker', 'beta']].reset_index(drop=True)
sortCoef = betas.sort('beta').reset_index(drop=True)
N = sortCoef.shape[0] / 10
# long_tickers = sortCoef.head(4*N).tail(3*N)[['ticker']].sort('ticker').reset_index(drop=True)
long_tickers = sortCoef.head(4 * N)[['ticker']].sort('ticker').reset_index(drop=True)
short_tickers = sortCoef.tail(4 * N)[['ticker']].sort('ticker').reset_index(drop=True)
beta_bound = 0.4
position_bound = 0.02
sector_bound = 0.02
three_factor_bound = 0.01
trim = 0.002

stats, wTickers = pf_optimizer(long_tickers, short_tickers, Coef, industry_data, CarhartDaily, StockBeta,
                               beta_bound, position_bound, sector_bound, three_factor_bound, trim,
                               build_start_date, build_end_date)

beta_distribution = pf_beta_distribution(wTickers, sortCoef)
print stats
print beta_distribution
# tmp_pf = long_tickers
# tmp_pf['weight'] = 1/np.float(tmp_pf.shape[0])
# print pf_beta_distribution(tmp_pf, sortCoef)

pf_value = pf_0208['position'].sum()
pf_new = wTickers.merge(StockPrices[StockPrices['date'] == build_end_date][['ticker', 'adj_close']]).reset_index(
    drop=True)
pf_new = pf_new.rename(columns={'adj_close': 'current price'})
pf_new['nShare'] = map(functions.intPart,
                       (pf_new['weight'] * pf_value / 2 / pf_new['current price']))
pf_new['value'] = pf_new['nShare'] * pf_new['current price']

pf_comp = pf_new.merge(pf_0208[['ticker', 'nShare', 'value']], on='ticker', how='outer',
                    suffixes = ('_new', '_old')).fillna(0).drop(['sector', 'weight'], axis=1)
rebalance_cost = np.abs(pf_comp['value_new'] - pf_comp['value_old']).sum() * 0.0015
print 'rebalance cost = %s' % rebalance_cost

pf_update = pf_comp[['ticker', 'nShare_old', 'nShare_new']].sort('ticker')
pf_update['change'] = pf_comp['nShare_new'] - pf_comp['nShare_old']
pf_update.to_csv('%s\Portfolio_Update_20150209.csv' % data_folder, index=False)