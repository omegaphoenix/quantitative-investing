# This code gets industry/sector data from Yahoo finance website

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import ReadData
import urllib2
import HTMLParser
from lxml import html
import requests

DataFolder = 'E:\Dropbox\CQA 2014\Data'
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
tickers = StockBeta[['ticker']]

# Remove '.' in the ticker
# tickers['ticker'] = tickers[['ticker']].apply(lambda s: s.replace('.', ''))

# Get data from Yahoo finance
industry_data = pd.DataFrame()
na_count = 0


def get_sector(ticker_old):
    na_count = 0
    ticker = ticker_old.replace('.', '')
    prefix = 'http://finance.yahoo.com/q/pr?s='
    # ticker = 'LMCK'
    response = urllib2.urlopen(prefix + ticker)
    html = response.read()
    response.close()
    # print html

    start_string = 'Sector:</td><td class="yfnc_tabledata1">'
    end_string = 'Full Time Employees'
    start_index = html.find(start_string)
    start_length = len(start_string)
    end_index = html.find(end_string)
    sub_string = html[start_index+start_length: end_index-1]

    if ticker == r'JW/A':
        sector = 'Services'
        industry = 'Publishing - Books'
    elif ticker == 'PGRE':
        sector = 'Financials'
        industry = ' Real Estate Development & Operations - NEC'
    elif start_index == -1 or end_index == -1 or sub_string[0:3] == 'N/A':
        na_count = 1
        sector = 'N/A'
        industry = 'N/A'
    else:
        sub_string = sub_string[sub_string.find('">')+2:]
        # print sub_string
        start_string_2 = 'Industry:</td><td class="yfnc_tabledata1">'
        start_index_2 = sub_string.find(start_string_2)
        start_length_2 = len(start_string_2)
        sub_string_2 = sub_string[(start_index_2+start_length_2):]
        sub_string_2 = sub_string_2[sub_string_2.find('">')+2:]
        # print sub_string

        h = HTMLParser.HTMLParser()
        sector = h.unescape(sub_string[0:sub_string.find('</a>')])
        industry = h.unescape(sub_string_2[0:sub_string_2.find('</a>')])
    print 'sector = %s' % sector
    print 'industry = %s ' % industry

    tmp_df = pd.DataFrame([{'ticker': ticker_old,
                            'sector': sector,
                            'industry': industry}])
    return tmp_df, na_count

for ticker_old in tickers['ticker']:
    print ticker_old
    (tmp_df, tmp_na_count) = get_sector(ticker_old)
    na_count += tmp_na_count
    industry_data = industry_data.append(tmp_df)

industry_data = industry_data.reset_index(drop=True)
industry_data = industry_data[['ticker', 'sector', 'industry']]
print 'N/A count = %s' % na_count
# We want to separate oil sector from other basic materials

for index, row in industry_data.iterrows():
    print row
    if 'Oil' in row[2]:
        row['sector'] = 'Oil'


# industry_tree = industry_data[['sector', 'industry']].drop_duplicates().sort(['sector', 'industry']).reset_index(drop=True)
industry_tree = pd.DataFrame({'count': industry_data[['sector', 'industry']].groupby(['sector', 'industry']).size()})\
    .reset_index()
industry_tree.to_csv(r'%s\sector_industry.csv' % DataFolder,
                     index=False)
print industry_tree
sector_tree = pd.DataFrame({'count': industry_data[['sector', 'industry']].groupby('sector').size()})\
    .reset_index().sort('count', ascending=False).reset_index(drop=True)
print sector_tree


industry_data.to_csv(r'%s\industry_data.csv' % DataFolder,
                     index=False)

def bar_sector(sector_tree):
    font_size = 15
    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)
    width = 0.4
    N = sector_tree.shape[0]
    ind = np.arange(N)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.bar(ind, sector_tree['count'])
    ax.set_xticks(ind + width)
    ax.set_xticklabels(sector_tree['sector'])
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=20 )
    plt.grid()
    plt.show()

# Ken's Sector/Industry data
industry_data_ken = pd.read_excel(r'%s\Russell 1000.xlsx' % DataFolder)
industry_data_ken = industry_data_ken[['Stripped ticker', 'Sector', 'Industry']].fillna('N/A')
industry_data_ken.columns = ['ticker', 'sector', 'industry']
# industry_data_ken_good = industry_data_ken.dropna(axis=0)
industry_data_ken_good = industry_data_ken[industry_data_ken['sector'] != 'N/A']
industry_data_ken_bad = industry_data_ken[industry_data_ken['sector'] == 'N/A']

industry_ken = pd.DataFrame({'count': industry_data_ken[['sector', 'industry']].groupby(['sector', 'industry']).size()})\
    .reset_index()
sector_ken = pd.DataFrame({'count': industry_data_ken[['sector', 'industry']].groupby('sector').size()})\
    .reset_index()
print sector_ken
bar_sector(sector_ken.sort('count', ascending=False))

# Now let's fill the blanks in industry_data_ken

appendant = pd.DataFrame()
for ticker in industry_data_ken_bad['ticker']:
    print ticker
    (tmp_df, tmp_na_count) = get_sector(ticker)
    appendant = appendant.append(tmp_df)
appendant_na = appendant[appendant['sector']=='N/A']
print appendant_na

appendant_tree = pd.DataFrame({'count': appendant[['sector', 'industry']].groupby(['sector', 'industry']).size()})\
    .reset_index()
appendant_sector_tree = pd.DataFrame({'count': appendant[['sector', 'industry']].groupby('sector').size()}).reset_index()
bar_sector(appendant_sector_tree.sort('count', ascending=False))

print industry_ken
print appendant_tree