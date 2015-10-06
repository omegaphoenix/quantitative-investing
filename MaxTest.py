#  This file tests an equal-weighted long/short portfolio
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import ReadData
import functions
# from dateutil import rrule
# from datetime import datetime

DataFolder = 'D:\Dropbox\CQA 2014\Data'
CAPITAL = 1e7
BUILD_DATE = datetime.datetime(2013, 11, 1)

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
StockReturns = ReadData.ReadReturn(DataFolder)
minDate = StockPrices['date'].min()
maxDate = StockPrices['date'].max()
tickers = ReadData.ReadTickers(DataFolder)
tickers = tickers[tickers['ticker'] != 'TDS'].reset_index(drop=True)

# dates_months = rrule.rrule(rrule.MONTHLY, dtstart=minDate, until=maxDate)

k = 1
dates = StockPrices['date'].astype(datetime.datetime).unique()
backDate = dates[1]
startDate = dates[20*k]
endDate = dates[20*(k+1)]

# For simplicity we only consider our stock universe, i.e. DataFrame tickers.
StockPrices = StockPrices.merge(tickers)

pfValues = pd.DataFrame()
for group in range(10):
    pfValues = pfValues.append(pd.DataFrame([
        {'date': startDate.date(),
         'value': CAPITAL,
         'MAX_group': group,
         'MAX_mean': 0}
    ]))
lastValues = np.ones(10) * CAPITAL
# lastValues.fill(CAPITAL)
while True:
    # print endDate.date()
    StockPricesSample = StockPrices[(StockPrices['date'] >= backDate) &
                                    (StockPrices['date'] <= endDate)]
    (StockMax, highMax, lowMax) = functions.findMax(StockReturns, startDate, tickers)
    # print 'backDate = %s' % backDate
    # print 'startDate = %s' % startDate
    print 'endDate = %s' % endDate
    for group in range(10):
        # print 'endDate: %s, group: %d' % (endDate, group)
        tmpGroup = StockMax[StockMax['MAX_group'] == group]
        MAX_mean = tmpGroup['MAX'].mean(axis=1)
        tmpTickers = tmpGroup[['ticker']]
        StockPricesSample2 = StockPricesSample.merge(tmpTickers)
        tmpGroup = tmpGroup.merge(
            StockPricesSample2[StockPricesSample2['date'] == startDate][['ticker', 'adj_close']]
            ).rename( columns={'adj_close': 'iniPrice'})
        N = tmpGroup.shape[0]
        # print 'N = %d' % N
        # tmpGroup['nShare'] = CAPITAL / N / tmpGroup['iniPrice']
        for i in range(20):
            tmpDate = StockPrices['date'].iloc[20*k + i + 1]
            # tmpPrice =
            tmpValue = tmpGroup.merge(StockPricesSample2[StockPricesSample2['date'] == tmpDate][['ticker', 'adj_close']])
            tmpValue['value'] = lastValues[group] / N / tmpValue['iniPrice'] * tmpValue['adj_close']
            # if tmpDate.date() == datetime.datetime(2005, 5, 16):
            #     print 'bug found!'
            #     test = tmpValue
            pfValues = pfValues.append(pd.DataFrame([
                {'date': tmpDate.date(),
                 'value': tmpValue['value'].sum(),
                 'MAX_group': group,
                 'MAX_mean': MAX_mean}
            ]))
        lastValues[group] = tmpValue['value'].sum()
        # print 'account value for group %d is %d' % (group, lastValues[group])
    k += 1
    if 20*(k+1) >= dates.size:
        break
    backDate = dates[20*(k-1)+1]
    startDate = dates[20*k]
    endDate = dates[20*(k+1)]

color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightblue', 'gold', 'pink']
pfDates = pd.DataFrame(pfValues['date'].unique()).astype(np.datetime64)
pfDates.columns = ['date']
tmpSP500 = SP500Prices.merge(pfDates)[['date', 'adj_close']]
iniSP500 = SP500Prices[SP500Prices['date']== StockPrices['date'].iloc[20]]['adj_close'].iloc[0]
tmpSP500['value'] = CAPITAL / iniSP500 * tmpSP500['adj_close']
for n in range(10):
    tmpPfValues = pfValues[pfValues['MAX_group'] == n ]
    tmpLabel = r'MAX group %d' % (n+1)
    plt.plot(pfDates, tmpPfValues['value'], label=tmpLabel, color=color_cycle[n], linewidth=2.0)
    # plt.legend('beta group %d' % (n+1))
plt.plot(pfDates, tmpSP500['value'], label = 'S&P500', linewidth=1.0)
plt.ylabel('Portfolio Value')
plt.legend(bbox_to_anchor=(0.2, 1))
plt.grid()
plt.show()

pfMAX = pfValues[['MAX_group', 'MAX_mean']].groupby('MAX_group').aggregate(np.mean)

# Equally weighted portfolio test