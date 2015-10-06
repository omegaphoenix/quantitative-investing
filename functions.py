import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime
import ReadData
import functions
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers, spmatrix
from matplotlib import pylab


def findMax(StockReturns, BUILD_DATE, tickers):
    # (StockPrices, SP500Prices, CarhartDaily, StockBeta) = ReadData.ReadAll(DataFolder)
    # StockReturns = ReadData.ReadReturn(DataFolder)

    dates = StockReturns[StockReturns['endDate'] <= BUILD_DATE]['endDate'].astype(datetime.datetime).unique()
    dates = dates[-20:]
    # print dates
    period_start = dates[0]
    # print 'MAX period start = %s' % period_start
    # print 'MAX period end = %s' % BUILD_DATE

    ReturnSample = StockReturns[(StockReturns['endDate'] >= period_start) &
                                (StockReturns['endDate'] <= BUILD_DATE)]
    StockMax = ReturnSample[['ticker', 'return']].sort('return', ascending=False).groupby('ticker').head(5).groupby(
        'ticker').agg(np.mean).reset_index()
    StockMax = StockMax.rename(columns={'return': 'MAX'}).sort('MAX', ascending=False).reset_index(drop=True)
    StockMax = StockMax.merge(tickers)
    N = StockMax.shape[0] / 10
    if (N < 30):
        raise Exception('Too few stocks!')
    highMax = StockMax.head(N).drop('MAX', axis=1)
    lowMax = StockMax.tail(4 * N).head(3 * N).drop('MAX', axis=1)
    StockMax['MAX_group'] = pd.qcut(StockMax['MAX'], 10, labels=False)
    # longTickers = lowMax['ticker']
    # shortTickers = highMax['ticker']
    # Now calculating weights
    highMax['pWeight'] = -1 / np.float64(highMax.shape[0])  # positive weights
    lowMax['nWeight'] = 1 / np.float64(lowMax.shape[0])  # negative weights
    return (StockMax, highMax, lowMax)


def iniWeights(tickers, longTickers, shortTickers):
    longTickers['pWeight'] = 1 / np.float64(longTickers.shape[0])  # positive weights
    shortTickers['nWeight'] = -1 / np.float64(longTickers.shape[0])  # negative weights
    wTickers = tickers.merge(longTickers, on='ticker', how='left')
    wTickers = wTickers.merge(shortTickers, on='ticker', how='left')
    wTickers = wTickers.fillna(0)
    wTickers['weight'] = wTickers['pWeight'] + wTickers['nWeight']
    wTickers = wTickers[['ticker', 'weight']]
    return wTickers


def testPortfolio(StockPrices, StockReturns, SP500Prices, StockBeta, wTickers, BUILD_DATE):
    # (highMax, lowMax) = findMax(StockReturns, BUILD_DATE, tickers)
    # wTickers = iniWeights(tickers, highMax, lowMax)
    # mCqaBeta = pd.merge(StockBeta, tickers).cqaBeta.astype(np.float64).as_matrix()
    BETA_BOUND = 0.5
    MAX_POSITION = 0.05
    CAPITAL = 1e7
    iniPrices = pd.merge(StockPrices[StockPrices.date == BUILD_DATE][['ticker', 'date', 'adj_close']],
                         wTickers[['ticker']], on='ticker').reset_index(drop=True)
    portfolio = pd.concat((wTickers[['ticker']], iniPrices[['date']], wTickers[['weight']], iniPrices[['adj_close']]),
                          axis=1)
    portfolio['nShare'] = ((portfolio['weight'] * CAPITAL / portfolio['adj_close']).map(intPart))
    portfolio['value'] = portfolio['nShare'] * portfolio['adj_close']
    # # Recording the initial portfolio
    iniPortfolio = portfolio
    resCapital = CAPITAL - sum(portfolio['value'])
    pfRecord = iniPortfolio
    pfValues = pd.DataFrame([{'date': BUILD_DATE, 'value': CAPITAL, 'beta_CQA': cqaBeta(StockBeta, wTickers)}, ])

    StockPricesTest = pd.merge(StockPrices[StockPrices.date > BUILD_DATE], wTickers[['ticker']], on='ticker')
    StockReturnsTest = pd.merge(StockReturns[StockReturns.endDate > BUILD_DATE], wTickers[['ticker']], on='ticker')
    testDates = StockReturnsTest.endDate.map(lambda t: t.date()).unique()

    # #testDates = testDates[:5]
    for date in testDates:
        print date
        tmpPrice = StockPricesTest[StockPricesTest.date == date][['ticker', 'adj_close']].reset_index(drop=True)
        tmpReturn = StockReturnsTest[StockReturnsTest.endDate == date].reset_index(drop=True)
        tmpData = pd.merge(portfolio.drop(['adj_close', 'date'], 1), tmpReturn, on='ticker')
        tmpData = pd.merge(tmpData, tmpPrice, on='ticker')
        tmpData['value'] = tmpData['value'] * tmpData['return'].map(math.exp)
        tmpValue = sum(tmpData['value']) + resCapital
        tmpData['weight'] = tmpData['value'] / tmpValue
        # tmpW = tmpData['weight'].as_matrix()
        # tmpSens = np.dot(tmpW.T, mCoef)
        tmpW = tmpData[['ticker', 'weight']]
        # tmpBeta = np.dot(tmpW, mCqaBeta)
        endDate = tmpReturn['endDate'].iget(0)
        # tmpBeta = cqaBeta(StockBeta, tmpW)
        tmpData['date'] = endDate
        tmpData = tmpData[['ticker', 'date', 'weight', 'adj_close', 'nShare', 'value']]
        portfolio = tmpData
        pfRecord = pfRecord.append(tmpData)
        pfValues = pfValues.append(pd.DataFrame([{'date': endDate,
                                                  'value': tmpValue,
                                                  'beta_CQA': cqaBeta(StockBeta, tmpW)}]))
        # 'beta_Mkt-RF' : tmpSens[0],
        # 'beta_SMB' : tmpSens[1],
        # 'beta_HML' : tmpSens[2],
        # 'beta_UMD' : tmpSens[3]},]))

    pfValues = pfValues.merge(SP500Prices[['adj_close', 'date']], on='date')
    pfValues = pfValues.rename(columns={'adj_close': 'SP500'})

    # summarize(pfValues, pfRecord, tickers, CAPITAL, BETA_BOUND)
    return (pfValues, pfRecord)


def testPortfolio2(StockPrices, SP500Prices, StockBeta, Coef, wTickers, total_value, sp500_value, BUILD_DATE, END_DATE):
    portfolio = wTickers.merge(
        StockPrices[StockPrices.date == BUILD_DATE][['ticker', 'adj_close']],
        on='ticker').reset_index(drop=True)
    ini_sp500 = SP500Prices[SP500Prices.date == BUILD_DATE]['adj_close'].iloc[0]
    portfolio['nShare'] = ((portfolio['weight'] * total_value / portfolio['adj_close']).map(intPart))
    portfolio['value'] = portfolio['nShare'] * portfolio['adj_close']
    res_capital = total_value - sum(portfolio['value'])
    # pfValues = pd.DataFrame([{'date': BUILD_DATE, 'value': total_value, 'beta_CQA': cqaBeta(StockBeta, wTickers)},])
    pfValues = pd.DataFrame()

    StockPricesTest = wTickers.merge(
        StockPrices[(StockPrices.date > BUILD_DATE) &
                    (StockPrices.date <= END_DATE)]).reset_index(drop=True)[['ticker', 'date', 'adj_close']]
    # test_dates = StockPricesTest['date'].map(lambda t: t.date()).unique()
    mBeta = Coef.merge(wTickers[['ticker']])['Mkt-RF'].as_matrix()
    mBeta_smb = Coef.merge(wTickers[['ticker']])['SMB'].as_matrix()
    mBeta_hml = Coef.merge(wTickers[['ticker']])['HML'].as_matrix()
    mBeta_umd = Coef.merge(wTickers[['ticker']])['UMD'].as_matrix()
    test_dates = StockPricesTest['date'].unique()

    # #testDates = testDates[:5]
    for date in test_dates:
        print date
        tmpPrice = StockPricesTest[StockPricesTest.date == date][['ticker', 'adj_close']].reset_index(drop=True)
        tmpData = tmpPrice.merge(portfolio.drop(['adj_close'], 1), on='ticker')
        # tmpData = StockPricesTest[StockPricesTest.date == date][['ticker', 'adj_close']].reset_index(drop=True)
        tmpData['value'] = tmpData['nShare'] * tmpData['adj_close']
        tmpValue = sum(tmpData['value']) + res_capital
        tmpData['weight'] = tmpData['value'] / tmpValue
        tmpW = tmpData[['ticker', 'weight']]
        # tmpSens = np.dot(tmpW['weight'].as_matrix().T, mCoef)
        # endDate = tmpReturn['endDate'].iget(0)
        # tmpBeta = cqaBeta(StockBeta, tmpW)
        # tmpData = tmpData[['ticker', 'date', 'weight', 'adj_close', 'nShare', 'value']]
        portfolio = tmpData
        pfValues = pfValues.append(pd.DataFrame([{'date': date,
                                                  'value': tmpValue,
                                                  'beta_CQA': cqaBeta(StockBeta, tmpW),
                                                  'beta_Mkt-RF': np.dot(tmpW['weight'].as_matrix().T, mBeta),
                                                  'beta_SMB': np.dot(tmpW['weight'].as_matrix().T, mBeta_smb),
                                                  'beta_HML': np.dot(tmpW['weight'].as_matrix().T, mBeta_hml),
                                                  'beta_UMD': np.dot(tmpW['weight'].as_matrix().T, mBeta_umd)
                                                 }]))
        # print 'size = %d' % pfValues.shape[0]

    pfValues = pfValues.merge(SP500Prices[['adj_close', 'date']], on='date')
    pfValues = pfValues.rename(columns={'adj_close': 'SP500'})
    pfValues['SP500value'] = sp500_value / ini_sp500 * pfValues['SP500']

    return pfValues


def testPortfolio3(StockPrices, SP500Prices, StockBeta, Coef, portfolio, total_value, sp500_value, build_date,
                   end_date):
    ini_sp500 = SP500Prices[SP500Prices.date == build_date]['adj_close'].iloc[0]
    # portfolio['nShare'] = ((portfolio['weight'] * total_value / portfolio['adj_close']).map(intPart))
    portfolio['value'] = portfolio['nShare'] * portfolio['price']
    wTickers = portfolio[['ticker']]
    # res_capital = total_value - sum(portfolio['value'])
    res_capital = total_value
    pfValues = pd.DataFrame()

    StockPricesTest = wTickers.merge(
        StockPrices[(StockPrices.date >= build_date) &
                    (StockPrices.date <= end_date)]). \
        reset_index(drop=True)[['ticker', 'date',
                                # 'adj_close']]
                                'close']].rename(columns={'close': 'adj_close'})
    StockPricesTest = StockPricesTest.sort('date')
    assert (StockPricesTest.shape[0] != 0)
    # test_dates = StockPricesTest['date'].map(lambda t: t.date()).unique()
    # mBeta = Coef.merge(wTickers[['ticker']])['Mkt-RF'].as_matrix()
    # mBeta_smb = Coef.merge(wTickers[['ticker']])['SMB'].as_matrix()
    # mBeta_hml = Coef.merge(wTickers[['ticker']])['HML'].as_matrix()
    # mBeta_umd = Coef.merge(wTickers[['ticker']])['UMD'].as_matrix()
    test_dates = StockPricesTest['date'].map(lambda t: t.date()).unique().astype(np.datetime64)

    # #testDates = testDates[:5]
    for date in test_dates:
        print date
        tmpPrice = StockPricesTest[StockPricesTest.date == date][['ticker', 'adj_close']].reset_index(drop=True)
        tmpData = tmpPrice.merge(portfolio.drop('value', 1), on='ticker')
        # print tmpData
        tmpData['value'] = tmpData['nShare'] * tmpData['adj_close']
        tmpValue = sum(tmpData['value']) + res_capital
        tmpData['weight'] = tmpData['value'] / tmpValue
        tmpW = tmpData[['ticker', 'weight']]
        # portfolio = tmpData
        pfValues = pfValues.append(pd.DataFrame([{'date': date,
                                                  'value': tmpValue,
                                                  'beta_CQA': cqaBeta(StockBeta, tmpW),
                                                  # 'beta_Mkt-RF' : np.dot(tmpW['weight'].as_matrix().T, mBeta),
                                                  # 'beta_SMB' : np.dot(tmpW['weight'].as_matrix().T, mBeta_smb),
                                                  # 'beta_HML' : np.dot(tmpW['weight'].as_matrix().T, mBeta_hml),
                                                  # 'beta_UMD' : np.dot(tmpW['weight'].as_matrix().T, mBeta_umd)
                                                 }]))
        # print 'size = %d' % pfValues.shape[0]

    pfValues = pfValues.merge(SP500Prices[['adj_close', 'date']], on='date')
    pfValues = pfValues.rename(columns={'adj_close': 'SP500'})
    pfValues['SP500value'] = sp500_value / ini_sp500 * pfValues['SP500']

    return pfValues


def testPortfolio_simple(StockPrices, StockReturns, SP500Prices,
                         StockBeta, portfolio, build_date, end_date=0):
    portfolio['value'] = portfolio['nShare'] * portfolio['price']
    portfolio['adj_close'] = portfolio['price']
    portfolio['date'] = build_date
    pfValues = pd.DataFrame([{'date': build_date, 'value': portfolio['value'].sum(),
                              # 'beta_CQA': cqaBeta(StockBeta, portfolio)
                             }, ])

    StockPricesTest = pd.merge(StockPrices[StockPrices.date > build_date], portfolio[['ticker']], on='ticker')
    StockReturnsTest = pd.merge(StockReturns[StockReturns.endDate > build_date], portfolio[['ticker']], on='ticker')
    testDates = StockReturnsTest.endDate.map(lambda t: t.date()).unique()

    # #testDates = testDates[:5]
    for date in testDates:
        print date
        tmpPrice = StockPricesTest[StockPricesTest.date == date][['ticker', 'adj_close']].reset_index(drop=True)
        tmpReturn = StockReturnsTest[StockReturnsTest.endDate == date].reset_index(drop=True)
        tmpData = pd.merge(portfolio.drop(['adj_close', 'date'], 1), tmpReturn, on='ticker')
        tmpData = pd.merge(tmpData, tmpPrice, on='ticker')
        tmpData['value'] = tmpData['value'] * tmpData['return'].map(math.exp)
        tmpValue = sum(tmpData['value'])
        tmpData['weight'] = tmpData['value'] / tmpValue
        # tmpW = tmpData[['ticker', 'weight']]
        endDate = tmpReturn['endDate'].iget(0)
        tmpData['date'] = endDate
        tmpData = tmpData[['ticker', 'date', 'weight', 'adj_close', 'nShare', 'value']]
        portfolio = tmpData
        # pfRecord = pfRecord.append(tmpData)
        pfValues = pfValues.append(pd.DataFrame([{'date': endDate,
                                                  'value': tmpValue,
                                                 }]))

    pfValues = pfValues.merge(SP500Prices[['adj_close', 'date']], on='date')
    pfValues = pfValues.rename(columns={'adj_close': 'SP500'})
    if end_date != 0:
        pfValues = pfValues[pfValues['date'] <= end_date]
    # return (pfValues, pfRecord)
    return pfValues


def summarize(pfValues, pfRecord, tickers, CAPITAL, BETA_BOUND):
    longPosition = pfRecord[pfRecord['value'] > 1][['date', 'value']].groupby('date').agg(np.sum).reset_index()
    # longPosition.columns = ['date', 'value']
    shortPosition = pfRecord[pfRecord['value'] < -1][['date', 'value']].groupby('date').agg(np.sum).reset_index()
    # shortPosition.columns = ['date', 'value']
    plt.figure(1)
    # plt.subplot(321)
    plt.subplot(221)
    plt.title('Account Value(r) vs S&P500(b)')
    plt.grid()
    plt.plot(pfValues.date, pfValues.value, 'r-', pfValues.date, pfValues.SP500 * CAPITAL / pfValues.SP500[0], 'b-')
    plt.axhline(CAPITAL)
    # plt.subplot(322)
    plt.subplot(222)
    plt.ylabel('beta_CQA')
    plt.grid()
    plt.plot(pfValues.date, pfValues.beta_CQA)
    plt.axhline(BETA_BOUND)
    plt.axhline(-BETA_BOUND)
    plt.subplot(223)
    plt.plot(longPosition.date, longPosition.value)
    plt.plot(shortPosition.date, shortPosition.value)
    plt.plot(longPosition.date, longPosition.value + shortPosition.value)
    plt.grid()
    plt.title('long/short position comparison')
    # plt.subplot(323)
    # plt.ylabel('beta_Mkt-RF')
    # plt.grid()
    # plt.plot(pfValues.date, pfValues['beta_Mkt-RF'])
    # plt.subplot(324)
    # plt.ylabel('beta_SMB')
    # plt.grid()
    # plt.plot(pfValues.date, pfValues.beta_SMB)
    # plt.subplot(325)
    # plt.ylabel('beta_HML')
    # plt.grid()
    # plt.plot(pfValues.date, pfValues.beta_HML)
    # plt.subplot(326)
    # plt.ylabel('beta_UMD')
    # plt.grid()
    # plt.plot(pfValues.date, pfValues.beta_UMD)
    plt.subplot(224)
    for ticker in tickers.ticker:
        print ticker
        tmpData = pfRecord[pfRecord.ticker == ticker]
        plt.plot(tmpData.date, tmpData.value)
    plt.show()


def intPart(x):
    return math.modf(x)[1]


def cqaBeta(StockBeta, w):
    mCqaBeta = pd.merge(StockBeta, w)[['weight', 'cqaBeta']].astype(np.float64).as_matrix()
    return np.dot(mCqaBeta[:, 0], mCqaBeta[:, 1])


def diffDates(date1, date2):
    return abs((date1 - date2).days)


def wReg(StockReturns, CarhartDaily, tickers, decay, START, END):
    StockReturns = StockReturns[(StockReturns['endDate'] >= START) &
                                (StockReturns['endDate'] <= END)].merge(tickers)
    Coef = pd.DataFrame()
    Res = pd.DataFrame()
    for ticker in StockReturns.ticker.unique():
        print "Getting regression coefficient for %s" % ticker
        tmpReturn = StockReturns[(StockReturns['ticker'] == ticker)]
        if not tmpReturn.empty:
            tmpData = tmpReturn.merge(CarhartDaily, left_on='endDate', right_on='date')
            # tmpData = tmpData.merge(SP500Returns, on = 'endDate')
            # tmpData['SP500-RF'] = tmpData['SP500Return']*100 - tmpData['RF']
            y = tmpData['return'] * 100 - tmpData['RF']
            X = sm.add_constant(tmpData[['Mkt-RF', 'SMB', 'HML', 'UMD']])
            w = tmpData['endDate'].map(lambda d: math.exp(-1 * decay * diffDates(d, END)))
            model = sm.WLS(y, X, weights=w).fit()
            tmpDF = pd.DataFrame(model.params).T
            tmpDF.rename(columns={'const': 'alphaFF'}, inplace=True)
            tmpDF['ticker'] = ticker
            # tmpRes = pd.concat((tmpData['endDate'], pd.DataFrame(model.resid, columns=[ticker])), axis=1)
            Coef = Coef.append(tmpDF)
            # Res = pd.concat((Res, tmpRes), axis=1, join='outer', join_axes=['endDate'])
            Res[ticker] = model.resid

    return (Coef, Res)


def GetBetas(StockReturns, CarhartDaily, decayList, start, end):
    betas = pd.DataFrame()
    for ticker in StockReturns.ticker.unique():
        print "Getting regression coefficient for %s" % ticker
        tmpReturn = StockReturns[(StockReturns['ticker'] == ticker)]
        if not tmpReturn.empty:
            tmpData = tmpReturn.merge(CarhartDaily, left_on='endDate', right_on='date')
            y = tmpData['return'] * 100 - tmpData['RF']
            X = sm.add_constant(tmpData[['Mkt-RF', 'SMB', 'HML', 'UMD']])
            tmpDF = pd.DataFrame()
            for decay in decayList:
                w = tmpData['endDate'].map(lambda d: math.exp(-1 * decay * diffDates(d, end)))
                model = sm.WLS(y, X, weights=w).fit()
                tmpDF['%.4f' % decay] = pd.DataFrame(model.params).T['Mkt-RF']
                # tmpDF.rename( columns = {'const' : '%.4f' % decay}, inplace= True)
            tmpDF['ticker'] = ticker
            betas = betas.append(tmpDF)
    return betas


def betaError(betaTrain, betaTest):
    diff = betaTrain - betaTest
    return np.linalg.norm(diff) / math.sqrt(diff.shape[0])


def GetDecay(StockReturns, CarhartDaily, decayList, START_TRAIN, END_TRAIN, START, END, tickers):
    StockReturnsTrain = StockReturns[(StockReturns['endDate'] >= START_TRAIN) &
                                     (StockReturns['endDate'] <= END_TRAIN)].merge(tickers)
    StockReturnsTest = StockReturns[StockReturns['endDate'] >= START].merge(tickers)

    decayList = [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]
    CoefTrain = GetBetas(StockReturnsTrain, CarhartDaily, decayList, START_TRAIN, END_TRAIN)
    noDecay = 0
    CoefTest = wReg(StockReturnsTest, CarhartDaily, noDecay, START, END).rename(columns={'Mkt-RF': 'betaTest'})
    errors = []
    for decay in decayList:
        tmpBeta = CoefTrain['%.4f' % decay]
        errors.append(betaError(tmpBeta, CoefTest['betaTest']))
    errors
    bestDecay = decayList[np.argmin(errors)]
    return bestDecay


# Linear programming, minimize beta exposure, no constraint on covariance
def pfOptimizer(longTickers, shortTickers, Coef, StockBeta, BETA_BOUND, WEIGHT_BOUND):
    longTickers['bWeight'] = 1  # binary weights
    shortTickers['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]), axis=0).sort('ticker').reset_index(
        drop=True)

    # sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers).rename(columns={'Mkt-RF': 'beta'})
    # control = matrix([1, 0.2, 0.2, 0.5])
    # scores = matrix(sens[['beta', 'SMB', 'HML', 'UMD']].as_matrix()) * control
    betas = Coef[['ticker', 'Mkt-RF']].merge(pfTickers).rename(columns={'Mkt-RF': 'beta'}).reset_index(drop=True)
    mBeta = matrix(betas['beta'])
    mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])

    longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex, 1) > 0.5])]).trans()
    # mLongIndex = matrix(np.diag(tickers.merge(longTickers, how='left').fillna(0)['bWeight']))
    shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex, 1) > 0.5])]).trans()
    # mShortIndex = matrix(np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight']))

    # wTickers = functions.iniWeights(pfTickers, shortTickers, longTickers) # initial weights
    wStart = matrix(functions.iniWeights(pfTickers, longTickers, shortTickers)['weight'])
    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N, 1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    # longBounds = matrix(np.ones((shortTickers.shape[0], 1)) * 0.002)
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    # shortBounds = matrix(np.ones((longTickers.shape[0], 1)) * (-0.005))
    A = matrix([[mCqaBeta], [-mCqaBeta],
                [longIndex], [-longIndex],
                [shortIndex], [-shortIndex],
                [id], [-id],
                [-mLongIndex], [mShortIndex]]).trans()
    b = matrix([BETA_BOUND, BETA_BOUND, 1, -0.98, -0.98, 1, wBounds, wBounds, longBounds, shortBounds])
    # A = matrix([ [longIndex], [-longIndex],
    # [shortIndex], [-shortIndex],
    # [id], [-id],
    #              [-mLongIndex], [mShortIndex]]).trans()
    # b = matrix([ 1, -0.98, -0.98, 1, wBounds, wBounds, longBounds, shortBounds])

    # scores = mBeta
    # sol = solvers.lp(-scores, A, b)
    sol = solvers.lp(-mBeta, A, b)
    w_res = sol['x']
    print 'cqaBeta = %.4f' % np.float64(w_res.trans() * mCqaBeta)[0, 0]
    print 'beta = %.4f' % np.float64(w_res.trans() * mBeta)[0, 0]
    wTickers = pfTickers
    wTickers['weight'] = w_res
    return wTickers


# optimizing sharpe ratio while restricting SMB, HML, UMD
def pfOptimizer2(longTickers, shortTickers, Coef, Res, CarhartDaily, StockBeta,
                 BETA_BOUND, WEIGHT_BOUND, back_date, start_date):
    longTickers['bWeight'] = 1  # binary weights
    shortTickers['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]),
                          axis=0).sort('ticker').reset_index(drop=True)

    sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers)
    mBeta = matrix(sens['Mkt-RF'])
    mBeta_smb = matrix(sens['SMB'])
    mBeta_hml = matrix(sens['HML'])
    mBeta_umd = matrix(sens['UMD'])
    mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])

    longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex, 1) > 0.5])]).trans()
    shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex, 1) > 0.5])]).trans()

    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N, 1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    total_cov = matrix(functions.get_cov(pfTickers, Coef, Res, CarhartDaily, back_date, start_date))
    q = matrix(np.zeros((pfTickers.shape[0], 1)))
    G = matrix([[mBeta], [-mBeta],
                [mBeta_smb], [-mBeta_smb],
                [mBeta_hml], [-mBeta_hml],
                [mBeta_umd], [-mBeta_umd],
                [mCqaBeta], [-mCqaBeta],
                [longIndex], [-longIndex],
                [shortIndex], [-shortIndex],
                [id], [-id],
                [-mLongIndex], [mShortIndex]]).trans()
    opt_comp = pd.DataFrame()
    max_slope = 0
    for k in range(11):
        print 'k = %d' % k
        h = matrix([(k + 1) * 0.05, -k * 0.05,
                    0.1, 0.1,
                    0.1, 0.1,
                    0.1, 0.1,
                    BETA_BOUND, BETA_BOUND,
                    1, -0.98,
                    -0.98, 1,
                    wBounds, wBounds,
                    longBounds, shortBounds])
        # sol = solvers.lp(-mBeta, A, b)
        try:
            sol = solvers.qp(total_cov, q, G, h)
        except ValueError:
            break
        w_res = sol['x']
        cov_res = sol['primal objective']
        beta_res = np.float64(w_res.trans() * mBeta)[0, 0]
        beta_cqa_res = np.float64(w_res.trans() * mCqaBeta)[0, 0]
        beta_smb_res = np.float64(w_res.trans() * mBeta_smb)[0, 0]
        beta_hml_res = np.float64(w_res.trans() * mBeta_hml)[0, 0]
        beta_umd_res = np.float64(w_res.trans() * mBeta_umd)[0, 0]
        opt_comp = opt_comp.append(pd.DataFrame([{'beta': beta_res,
                                                  'beta_cqa': beta_cqa_res,
                                                  'beta_smb': beta_smb_res,
                                                  'beta_hml': beta_hml_res,
                                                  'beta_umd': beta_umd_res,
                                                  'vol': math.sqrt(cov_res),
                                                 }]))
        slope = beta_res / math.sqrt(cov_res)
        if slope > max_slope:
            w_final = w_res
    wTickers = pfTickers
    wTickers['weight'] = w_final
    return (opt_comp, wTickers)


# optimizing sharpe ratio while NOT restricting SMB, HML, UMD
def pfOptimizer3(longTickers, shortTickers, Coef, Res, CarhartDaily, StockBeta,
                 BETA_BOUND, WEIGHT_BOUND, back_date, start_date):
    longTickers['bWeight'] = 1  # binary weights
    shortTickers['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]),
                          axis=0).sort('ticker').reset_index(drop=True)

    sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers)
    mBeta = matrix(sens['Mkt-RF'])
    # mBeta_smb = matrix(sens['SMB'])
    # mBeta_hml = matrix(sens['HML'])
    # mBeta_umd = matrix(sens['UMD'])
    mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])

    longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex, 1) > 0.5])]).trans()
    shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex, 1) > 0.5])]).trans()

    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N, 1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    total_cov = matrix(functions.get_cov(pfTickers, Coef, Res, CarhartDaily, back_date, start_date))
    q = matrix(np.zeros((pfTickers.shape[0], 1)))
    G = matrix([[mBeta], [-mBeta],
                # [mBeta_smb], [-mBeta_smb],
                # [mBeta_hml], [-mBeta_hml],
                # [mBeta_umd], [-mBeta_umd],
                [mCqaBeta], [-mCqaBeta],
                [longIndex], [-longIndex],
                [shortIndex], [-shortIndex],
                [id], [-id],
                [-mLongIndex], [mShortIndex]]).trans()
    opt_comp = pd.DataFrame()
    max_slope = 0
    for k in range(11):
        print 'k = %d' % k
        h = matrix([(k + 1) * 0.05, -k * 0.05,
                    # 0.1, 0.1,
                    # 0.1, 0.1,
                    # 0.1, 0.1,
                    BETA_BOUND, BETA_BOUND,
                    1, -0.98,
                    -0.98, 1,
                    wBounds, wBounds,
                    longBounds, shortBounds])
        # sol = solvers.lp(-mBeta, A, b)
        try:
            sol = solvers.qp(total_cov, q, G, h)
        except ValueError:
            break
        w_res = sol['x']
        cov_res = sol['primal objective']
        beta_res = np.float64(w_res.trans() * mBeta)[0, 0]
        beta_cqa_res = np.float64(w_res.trans() * mCqaBeta)[0, 0]
        # beta_smb_res = np.float64(w_res.trans() * mBeta_smb)[0,0]
        # beta_hml_res = np.float64(w_res.trans() * mBeta_hml)[0,0]
        # beta_umd_res = np.float64(w_res.trans() * mBeta_umd)[0,0]
        opt_comp = opt_comp.append(pd.DataFrame([{'beta': beta_res,
                                                  'beta_cqa': beta_cqa_res,
                                                  # 'beta_smb': beta_smb_res,
                                                  # 'beta_hml': beta_hml_res,
                                                  # 'beta_umd': beta_umd_res,
                                                  'vol': math.sqrt(cov_res),
                                                 }]))
        slope = beta_res / math.sqrt(cov_res)
        if slope > max_slope:
            w_final = w_res
    wTickers = pfTickers
    wTickers['weight'] = w_final
    return (opt_comp, wTickers)


# control beta, minimizing covariance matrix
def pfOptimizer4(longTickers, shortTickersHigh, shortTickersLow,
                 Coef, Res, CarhartDaily, StockBeta,
                 BETA_BOUND, WEIGHT_BOUND, back_date, start_date):
    shortTickers = shortTickersHigh.append(shortTickersLow).sort('ticker').reset_index(drop=True)
    longTickers['bWeight'] = 1  # binary weights
    shortTickers['bWeight'] = -1
    shortTickersLow['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]),
                          axis=0).sort('ticker').reset_index(drop=True)

    sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers)
    mBeta = matrix(sens['Mkt-RF'])
    mBeta_smb = matrix(sens['SMB'])
    mBeta_hml = matrix(sens['HML'])
    mBeta_umd = matrix(sens['UMD'])
    mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])

    longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex, 1) > 0.5])]).trans()
    shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex, 1) > 0.5])]).trans()
    shortLowIndex = -matrix(pfTickers.merge(shortTickersLow, how='left').fillna(0)['bWeight'])

    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N, 1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    total_cov = matrix(functions.get_cov(pfTickers, Coef, Res, CarhartDaily, back_date, start_date))
    q = matrix(np.zeros((pfTickers.shape[0], 1)))
    G = matrix([[mBeta], [-mBeta],
                [mBeta_smb], [-mBeta_smb],
                [mBeta_hml], [-mBeta_hml],
                [mBeta_umd], [-mBeta_umd],
                [mCqaBeta], [-mCqaBeta],
                [longIndex], [-longIndex],
                [shortIndex], [-shortIndex],
                [shortLowIndex], [-shortLowIndex],
                [id], [-id],
                [-mLongIndex], [mShortIndex]]).trans()
    opt_comp = pd.DataFrame()
    h = matrix([0.1, 0.2,
                0.1, 0.1,
                0.1, 0.1,
                0.1, 0.1,
                BETA_BOUND, BETA_BOUND,
                1, -0.98,
                -0.98, 1,
                0.4, 0,
                wBounds, wBounds,
                longBounds, shortBounds])
    sol = solvers.qp(total_cov, q, G, h)
    w_final = sol['x']
    wTickers = pfTickers
    wTickers['weight'] = w_final
    return (opt_comp, wTickers)


def pfOptimizer5(longTickers, shortTickersHigh, shortTickersLow,
                 Coef, Res, CarhartDaily, StockBeta,
                 BETA_BOUND, WEIGHT_BOUND, back_date, start_date):
    shortTickers = shortTickersHigh.append(shortTickersLow).sort('ticker').reset_index(drop=True)
    longTickers['bWeight'] = 1  # binary weights
    shortTickers['bWeight'] = -1
    shortTickersLow['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]),
                          axis=0).sort('ticker').reset_index(drop=True)

    sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers)
    mBeta = matrix(sens['Mkt-RF'])
    mBeta_smb = matrix(sens['SMB'])
    mBeta_hml = matrix(sens['HML'])
    mBeta_umd = matrix(sens['UMD'])
    mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])

    longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex, 1) > 0.5])]).trans()
    shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex, 1) > 0.5])]).trans()
    shortLowIndex = -matrix(pfTickers.merge(shortTickersLow, how='left').fillna(0)['bWeight'])

    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N, 1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    total_cov = matrix(functions.get_cov(pfTickers, Coef, Res, CarhartDaily, back_date, start_date))
    q = matrix(np.zeros((pfTickers.shape[0], 1)))
    G = matrix([[mBeta], [-mBeta],
                [mBeta_smb], [-mBeta_smb],
                [mBeta_hml], [-mBeta_hml],
                [mBeta_umd], [-mBeta_umd],
                [mCqaBeta], [-mCqaBeta],
                [longIndex], [-longIndex],
                [shortIndex], [-shortIndex],
                [-shortLowIndex],
                [id], [-id],
                [-mLongIndex], [mShortIndex]]).trans()
    opt_comp = pd.DataFrame()
    h = matrix([0.1, 0.2,
                0.1, 0.1,
                0.1, 0.1,
                0.1, 0.1,
                BETA_BOUND, BETA_BOUND,
                1, -0.98,
                -0.98, 1,
                0.3,
                wBounds, wBounds,
                longBounds, shortBounds])
    sol = solvers.qp(total_cov, q, G, h)
    w_final = sol['x']
    wTickers = pfTickers
    wTickers['weight'] = w_final
    return (opt_comp, wTickers)


def pfOptimizer6(longTickers, shortTickersHigh, shortTickersLow,
                 Coef, Res, CarhartDaily, StockBeta,
                 BETA_BOUND, WEIGHT_BOUND, back_date, start_date):
    shortTickers = shortTickersHigh.append(shortTickersLow).sort('ticker').reset_index(drop=True)
    longTickers['bWeight'] = 1  # binary weights
    shortTickers['bWeight'] = -1
    shortTickersLow['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]),
                          axis=0).sort('ticker').reset_index(drop=True)

    sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers)
    mBeta = matrix(sens['Mkt-RF'])
    mBeta_smb = matrix(sens['SMB'])
    mBeta_hml = matrix(sens['HML'])
    mBeta_umd = matrix(sens['UMD'])
    mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])

    longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex, 1) > 0.5])]).trans()
    shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex, 1) > 0.5])]).trans()
    shortLowIndex = -matrix(pfTickers.merge(shortTickersLow, how='left').fillna(0)['bWeight'])

    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N, 1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    total_cov = matrix(functions.get_cov(pfTickers, Coef, Res, CarhartDaily, back_date, start_date))
    q = matrix(np.zeros((pfTickers.shape[0], 1)))
    G = matrix([[mBeta], [-mBeta],
                # [mBeta_smb], [-mBeta_smb],
                # [mBeta_hml], [-mBeta_hml],
                # [mBeta_umd], [-mBeta_umd],
                [mCqaBeta], [-mCqaBeta],
                [longIndex], [-longIndex],
                [shortIndex], [-shortIndex],
                [-shortLowIndex],
                [id], [-id],
                [-mLongIndex], [mShortIndex]]).trans()
    opt_comp = pd.DataFrame()
    h = matrix([0.1, 0.2,
                # 0.1, 0.1,
                # 0.1, 0.1,
                # 0.1, 0.1,
                BETA_BOUND, BETA_BOUND,
                1, -0.98,
                -0.98, 1,
                0.3,
                wBounds, wBounds,
                longBounds, shortBounds])
    sol = solvers.qp(total_cov, q, G, h)
    w_final = sol['x']
    wTickers = pfTickers
    wTickers['weight'] = w_final
    return (opt_comp, wTickers)


def pfOptimizer7(longTickers, shortTickersHigh, shortTickersLow,
                 Coef, Res, CarhartDaily, StockBeta,
                 BETA_BOUND, WEIGHT_BOUND, back_date, start_date):
    shortTickers = shortTickersHigh.append(shortTickersLow).sort('ticker').reset_index(drop=True)
    longTickers['bWeight'] = 1  # binary weights
    shortTickers['bWeight'] = -1
    shortTickersLow['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]),
                          axis=0).sort('ticker').reset_index(drop=True)

    sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers)
    mBeta = matrix(sens['Mkt-RF'])
    mBeta_smb = matrix(sens['SMB'])
    mBeta_hml = matrix(sens['HML'])
    mBeta_umd = matrix(sens['UMD'])
    mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])

    longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex, 1) > 0.5])]).trans()
    shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex, 1) > 0.5])]).trans()
    shortLowIndex = -matrix(pfTickers.merge(shortTickersLow, how='left').fillna(0)['bWeight'])

    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N, 1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    # total_cov = matrix(functions.get_cov(pfTickers, Coef, Res, CarhartDaily, back_date, start_date))
    # q = matrix(np.zeros((pfTickers.shape[0], 1)))
    q = -mBeta_smb - mBeta_hml
    G = matrix([[mBeta], [-mBeta],
                # [mBeta_smb], [-mBeta_smb],
                # [mBeta_hml], [-mBeta_hml],
                [mBeta_umd], [-mBeta_umd],
                [mCqaBeta], [-mCqaBeta],
                [longIndex], [-longIndex],
                [shortIndex], [-shortIndex],
                [-shortLowIndex],
                [id], [-id],
                [-mLongIndex], [mShortIndex]]).trans()
    opt_comp = pd.DataFrame()
    h = matrix([0.1, 0.2,
                # 0.1, 0.1,
                # 0.1, 0.1,
                0.1, 0.1,
                BETA_BOUND, BETA_BOUND,
                1, -0.98,
                -0.98, 1,
                0.3,
                wBounds, wBounds,
                longBounds, shortBounds])
    sol = solvers.lp(q, G, h)
    w_final = sol['x']
    wTickers = pfTickers
    wTickers['weight'] = w_final
    return (opt_comp, wTickers)


def pfOptimizer8(longTickers, shortTickersHigh, shortTickersLow,
                 Coef, Res, CarhartDaily, StockBeta,
                 BETA_BOUND, WEIGHT_BOUND, back_date, start_date,
                 diff_mean, diff_threshold):
    shortTickers = shortTickersHigh.append(shortTickersLow).sort('ticker').reset_index(drop=True)
    longTickers['bWeight'] = 1  # binary weights
    shortTickers['bWeight'] = -1
    shortTickersLow['bWeight'] = -1
    pfTickers = pd.concat((shortTickers[['ticker']], longTickers[['ticker']]),
                          axis=0).sort('ticker').reset_index(drop=True)

    sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(pfTickers)
    mBeta = matrix(sens['Mkt-RF'])
    mBeta_smb = matrix(sens['SMB'])
    mBeta_hml = matrix(sens['HML'])
    mBeta_umd = matrix(sens['UMD'])
    mCqaBeta = matrix(StockBeta.merge(pfTickers)['cqaBeta'])

    longIndex = matrix(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = np.diag(pfTickers.merge(longTickers, how='left').fillna(0)['bWeight'])
    mLongIndex = matrix(mLongIndex[np.logical_or.reduce([np.sum(mLongIndex, 1) > 0.5])]).trans()
    shortIndex = -matrix(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = -np.diag(pfTickers.merge(shortTickers, how='left').fillna(0)['bWeight'])
    mShortIndex = matrix(mShortIndex[np.logical_or.reduce([np.sum(mShortIndex, 1) > 0.5])]).trans()
    shortLowIndex = -matrix(pfTickers.merge(shortTickersLow, how='left').fillna(0)['bWeight'])

    N = pfTickers.shape[0]
    id = spmatrix(1.0, range(N), range(N))
    wBounds = matrix(np.ones((N, 1)) * WEIGHT_BOUND)
    longBounds = matrix(np.zeros((shortTickers.shape[0], 1)))
    shortBounds = matrix(np.zeros((longTickers.shape[0], 1)))
    if abs(diff_mean) < diff_threshold:
        total_cov = matrix(functions.get_cov(pfTickers, Coef, Res, CarhartDaily, back_date, start_date))
        q = matrix(np.zeros((pfTickers.shape[0], 1)))
        G = matrix([[mBeta], [-mBeta],
                    [mBeta_smb], [-mBeta_smb],
                    [mBeta_hml], [-mBeta_hml],
                    [mBeta_umd], [-mBeta_umd],
                    [mCqaBeta], [-mCqaBeta],
                    [longIndex], [-longIndex],
                    [shortIndex], [-shortIndex],
                    [-shortLowIndex],
                    [id], [-id],
                    [-mLongIndex], [mShortIndex]]).trans()
        opt_comp = pd.DataFrame()
        h = matrix([0.1, 0.2,
                    0.1, 0.1,
                    0.1, 0.1,
                    0.1, 0.1,
                    BETA_BOUND, BETA_BOUND,
                    1, -0.98,
                    -0.98, 1,
                    0.3,
                    wBounds, wBounds,
                    longBounds, shortBounds])
        sol = solvers.qp(total_cov, q, G, h)
    else:
        if diff_mean >= diff_threshold:
            q = mBeta_smb
        elif diff_mean <= -diff_threshold:
            q = -mBeta_smb
        G = matrix([[mBeta], [-mBeta],
                    # [mBeta_smb], [-mBeta_smb],
                    [mBeta_hml], [-mBeta_hml],
                    [mBeta_umd], [-mBeta_umd],
                    [mCqaBeta], [-mCqaBeta],
                    [longIndex], [-longIndex],
                    [shortIndex], [-shortIndex],
                    [-shortLowIndex],
                    [id], [-id],
                    [-mLongIndex], [mShortIndex]]).trans()
        opt_comp = pd.DataFrame()
        h = matrix([0.1, 0.2,
                    # 0.1, 0.1,
                    0.1, 0.1,
                    0.1, 0.1,
                    BETA_BOUND, BETA_BOUND,
                    1, -0.98,
                    -0.98, 1,
                    0.3,
                    wBounds, wBounds,
                    longBounds, shortBounds])
        sol = solvers.lp(q, G, h)
    w_final = sol['x']
    wTickers = pfTickers
    wTickers['weight'] = w_final
    return (opt_comp, wTickers)


def get_cov(tickers, Coef, Res, CarhartDaily, start_date, end_date):
    CarhartSample = CarhartDaily[(CarhartDaily['date'] >= start_date) &
                                 (CarhartDaily['date'] <= end_date)][['Mkt-RF', 'SMB', 'HML', 'UMD']].reset_index(
        drop=True)
    ResSample = Res[tickers.ticker]
    FactorCov = np.cov(CarhartSample.as_matrix().transpose())
    ResCov = np.diag(np.diag(np.cov(ResSample.as_matrix().transpose())))
    sens = Coef[['ticker', 'Mkt-RF', 'SMB', 'HML', 'UMD']].merge(tickers[['ticker']]).sort('ticker').drop('ticker',
                                                                                                          axis=1).as_matrix()

    total_cov = np.dot(np.dot(sens, FactorCov), sens.transpose()) + ResCov
    return total_cov / 100


def moving_average(date_w_return, window_size):
    ave_res = pd.DataFrame()
    for k in range(window_size, date_w_return.shape[0]):
        ave_res = ave_res.append(pd.DataFrame([{
                                                   'date': date_w_return.iloc[k - 1]['date'],
                                                   'ave_return': date_w_return.iloc[k - window_size: k - 1][
                                                       'return'].mean()
                                               }]))
    return ave_res


def plt_moving_average(date_w_return, factor_name):
    ave_return_20 = functions.moving_average(date_w_return, 20)
    ave_return_40 = functions.moving_average(date_w_return, 40)
    ave_return_130 = functions.moving_average(date_w_return, 130)
    ave_return_200 = functions.moving_average(date_w_return, 200)
    plt.plot(date_w_return['date'], date_w_return['return'], color='lightblue')
    # plt.plot(ave_return_20['date'], ave_return_20['ave_return'], color='g', linewidth=2.0)
    plt.plot(ave_return_40['date'], ave_return_40['ave_return'], label='40 day average', color='r', linewidth=2.0)
    plt.plot(ave_return_130['date'], ave_return_130['ave_return'], label='130 day average', color='blue', linewidth=2.0)
    # plt.plot(ave_return_200['date'], ave_return_200['ave_return'], color='pink', linewidth=2.0)
    plt.title('%s factor return over time' % factor_name)
    plt.grid()
    plt.legend(bbox_to_anchor=(0.2, 1))
    pylab.ylim([-1, 1])
    plt.show()

