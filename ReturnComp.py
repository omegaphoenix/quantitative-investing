import math
import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import datetime
import ReadData
from scipy.optimize import minimize

DataFolder = 'E:\Dropbox\CQA 2014\Data'

(StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
SP500Returns = ReadData.GetSP500Return(SP500Prices)
returnComp = SP500Returns.merge(CarhartDaily[['date', 'Mkt-RF']], on = 'date')

returnComp['SP500Return'] = returnComp['SP500Return'] * 100
returnCompSort = returnComp.sort('SP500Return')

def comp(returnCompSort):
    plt.plot(returnCompSort['SP500Return'], 'r-')
    plt.plot(returnCompSort['Mkt-RF'], 'b-')
    plt.show()

comp(returnComp.tail(100).sort('SP500Return'))
