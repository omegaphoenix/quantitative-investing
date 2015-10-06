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

DataFolder = 'D:\Dropbox\CQA 2014\Data'
(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
account_value = pd.read_csv(r'%s/AccountValues.csv' % DataFolder, parse_dates=[0])
CAPITAL = 1e7

start_date = datetime.datetime(2014, 11, 10)
end_date = datetime.datetime(2014, 12, 16)

account_value = account_value[(account_value['Date'] >= start_date) &
                              (account_value['Date'] <= end_date)].ix[:, 0:2]
account_value.columns = ['date', 'value']
account_value['value'] = account_value['value'].map(lambda s: float(s.replace(',','')[1:]))

sp500_value = SP500Prices[(SP500Prices['date'] >= start_date) &
                              (SP500Prices['date'] <= end_date)]
ini_sp500 = sp500_value['adj_close'].iloc[0]
sp500_value['sp500value'] = sp500_value['adj_close'] / ini_sp500 * CAPITAL
account_value = account_value.merge(sp500_value, on='date')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(account_value['date'], account_value['value'], color='b', linewidth=2)
# ax.plot(account_value['date'], account_value['sp500value'], color='r', linewidth=2)
# ax.grid()
# plt.show()

plt.figure()
matplotlib.rc('xtick', labelsize=24)
matplotlib.rc('ytick', labelsize=24)
plt.plot(account_value['date'], account_value['value'], color='b', linewidth=4,
         label='Portfolio Value')
plt.plot(account_value['date'], account_value['sp500value'], color='r', linewidth=4,
         label='S&P 500')
# plt.yticks()
# ax.Tick(size=14)
plt.legend(loc=3, prop={'size': 36})
plt.grid()
plt.show()


