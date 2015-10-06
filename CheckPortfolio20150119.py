# This file checks which positions disappeared in our portfolio
# This file also checks the current weights in each sector
import pandas as pd
import numpy as np
import matplotlib
import ReadData
import functions
import datetime
import math
import matplotlib.pyplot as plt


data_folder = 'E:/Dropbox/CQA 2014/Data'

pf_1110 = pd.read_csv('%s/Portfolio_20141110.csv' % data_folder).replace('BF.B', 'BFB')
pf_0119 = pd.read_csv('%s/OpenPositions_20150119.csv' % data_folder)
pf_0119 = pf_0119[['Symbol', 'QTY', 'Price Paid', 'Last Price', 'Market Value']]\
    .rename(columns={'Symbol': 'ticker',
                     'QTY': 'nShare',
                     'Price Paid': 'price',
                     'Last Price': 'price_current',
                     'Market Value': 'value'})\
    .sort('ticker').reset_index(drop=True)
dollar_to_float = lambda s: float(s.strip('$').replace(',', ''))
pf_0119['price'] = pf_0119['price'].map(dollar_to_float)
pf_0119['price_current'] = pf_0119['price_current'].map(dollar_to_float)
pf_0119['value'] = pf_0119['value'].map(dollar_to_float) * pf_0119['nShare'].map(lambda q: np.sign(q))


# Find out which tickers are missing
def missing_ticker():
    pf_comp = pf_1110.merge(pf_0119, on='ticker', how='left')
    print pf_comp

    pf_missing = pf_comp[pf_comp.isnull().any(axis=1)]
    print pf_missing

# Now let's look at the sector weights
ticker_sector = pd.read_csv('%s/industry_data.csv' % data_folder)
for index, row in ticker_sector.iterrows():
    # print row
    if not pd.isnull(row[2]) and 'Oil' in row[2]:
        row['sector'] = 'Oil'

# print ticker_sector[ticker_sector['ticker'] == 'BFB'] # it's 'BFB'
# print ticker_sector[ticker_sector['ticker'] == 'BF.B']

sector_list = ticker_sector['sector'].unique().tolist()
pf_0119 = pf_0119.merge(ticker_sector[['ticker', 'sector']], on='ticker', how='left')
pf_1110 = pf_1110.merge(ticker_sector[['ticker', 'sector']], on='ticker', how='left')
# pf_0119_sector_na = pf_0119[pf_0119.isnull().any(axis=1)]
# print pf_0119_sector_na

sector_summary = pd.DataFrame()
for sector in sector_list:
    print sector
    tmp_pf = pf_0119[pf_0119['sector'] == sector]
    sector_long = tmp_pf['value'].map(lambda v: max(v, 0)).sum()
    sector_short = tmp_pf['value'].map(lambda v: min(v, 0)).sum()
    tmp_df = pd.DataFrame([{'sector': sector,
                            'long_value': sector_long,
                            'short_value': sector_short}])
    sector_summary = sector_summary.append(tmp_df)
sector_summary = sector_summary[['sector', 'long_value', 'short_value']].reset_index(drop=True)
sector_summary['net_value'] = sector_summary['long_value'] + sector_summary['short_value']
sector_summary['long_weight'] = sector_summary['long_value']\
    .map(lambda s: str(round(s/sector_summary['long_value'].sum()*100, 2)) + '%')
sector_summary['short_weight'] = sector_summary['short_value']\
    .map(lambda s: str(round(s/sector_summary['short_value'].sum()*100, 2)) + '%')
pf_value = sector_summary['long_value'].sum() - sector_summary['short_value'].sum()
sector_summary['net_weight'] = sector_summary['net_value']\
    .map(lambda s: str(round(s/pf_value*100, 2)) + '%')

labels = sector_summary['sector'].tolist()
color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightblue', 'gold', 'pink']
def sector_pie_chart():
    plt.subplot(121)
    plt.title('long components')
    plt.pie(sector_summary['long_value'], labels=labels, colors=color_cycle)
    plt.subplot(122)
    plt.title('short components')
    plt.pie(abs(sector_summary['short_value']), labels=labels, colors=color_cycle)
    plt.show()

def sector_bar_chart():
    n_groups = sector_summary.shape[0]
    font_size = 15
    # matplotlib.rc('xtick', labelsize=font_size)
    # matplotlib.rc('ytick', labelsize=font_size)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, sector_summary['long_value'], bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Long Position')

    rects2 = plt.bar(index, sector_summary['short_value'], bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='Short Position')
    rects2 = plt.bar(index + bar_width, sector_summary['net_value'], bar_width,
                     alpha=opacity,
                     color='g',
                     error_kw=error_config,
                     label='Net Position')
    plt.xlabel('Sector', size=font_size)
    plt.ylabel('Market Value', size=font_size)
    plt.title('Long/Shoart Market Value for each Sector', size=font_size)
    plt.xticks(index + bar_width, sector_list, rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

sector_bar_chart()

print sector_summary[['sector', 'long_weight', 'short_weight', 'net_weight']]

# Now let's look at sector returns since 11/10

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(data_folder)
StockReturns = ReadData.ReadReturn(data_folder)
build_date = datetime.datetime(2014, 11, 10)
end_date = StockPrices['date'].max()
end_tickers = pd.DataFrame({'ticker': StockPrices[StockPrices['date'] == end_date]['ticker'].unique()})
sector_values = pd.DataFrame()
for sector in sector_list:
    # sector = sector_list[0]
    wTickers = ticker_sector[ticker_sector['sector'] == sector].merge(end_tickers)
    wTickers['weight'] = 1/float(wTickers.shape[0])
    (pfValues, pfRecord) = functions.testPortfolio(StockPrices, StockReturns, SP500Prices, StockBeta, wTickers, build_date)
    tmp_df = pfValues[['date', 'value']].rename(columns={'value': sector})
    if sector_values.empty:
        sector_values = tmp_df
    else:
        sector_values = sector_values.merge(tmp_df)


def plot_sector_returns(sector_values):
    color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightblue', 'gold', 'pink']
    n = 0
    font_size = 15
    for sector in sector_list:
        plt.plot(sector_values['date'], sector_values[sector], label=sector, color=color_cycle[n], linewidth=3.0)
        # plt.legend('beta group %d' % (n+1))
        n += 1
    plt.ylabel('Sector Portfolio Value', size=font_size)
    plt.legend(loc=3, prop={'size': font_size})
    plt.grid()
    plt.show()


# # Now let's look at sector returns in our portfolio
# sector_values = pd.DataFrame()
# for sector in sector_list:
#     # sector = sector_list[0]
#     wTickers = pf_0119[pf_0119['sector'] == sector].merge(end_tickers)
#     ini_value = wTickers['value'].sum()
#     wTickers = wTickers.drop('value',1)
#     # wTickers['weight'] = 1/float(wTickers.shape[0])
#     pfValues = testPortfolio_simple(StockPrices, StockReturns, SP500Prices, StockBeta, wTickers, build_date)
#     # (pfValues, pfRecord) = functions.testPortfolio_simple(StockPrices, StockReturns, SP500Prices, StockBeta, wTickers, build_date)
#     tmp_df = pfValues[['date', 'value']].rename(columns={'value': sector})
#     tmp_df[sector] = tmp_df[sector]/1e7*ini_value
#     if sector_values.empty:
#         sector_values = tmp_df
#     else:
#         sector_values = sector_values.merge(tmp_df)
# sector_values['total_value'] = sector_values.sum(axis=1)
#
# plot_sector_returns(sector_values)
