import urllib2
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import csv

def pullData(stock):
    try:
        print 'Currently pulling', stock
        print str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        urlToVisit = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=2m/csv'
        saveFileLine = 'Text Files/'+stock+'.txt'

        try:#Check for existing file and figure out where to add from.
            readExistingData = open(saveFileLine,'r').read()
            splitExisting = readExistingData.split('\n')
            mostRecentLine = splitExisting[-2]
            lastDate = mostRecentLine.split(',')[0]
        except Exception,e:#Otherwise a new file will be created.
            print str(e)
            time.sleep(1)
            lastDate = 0

        saveFile = open(saveFileLine,'a')
        sourceCode = urllib2.urlopen(urlToVisit).read()
        splitSource = sourceCode.split('\n')

        for eachLine in splitSource:
            if 'values' not in eachLine:
                splitLine = eachLine.split(',')
                if len(splitLine)==6:
                    if int(splitLine[0]) > int(lastDate):
                        lineToWrite = eachLine+'\n'
                        saveFile.write(lineToWrite)

        saveFile.close()

        print 'Pulled', stock
        print 'sleeping....'
        print str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        time.sleep(1)

    except Exception,e:
        print 'main loop',str(e)

def graphData(stock):
    try:
        stockFile = 'Text Files/'+stock+'.txt'
        date, closep, highp, lowp, openp, volume = np.loadtxt(stockFile,delimiter=',',unpack=True,converters={ 0: mdates.strpdate2num('%Y%m%d')})
        fig = plt.figure()
        ax1 = plt.subplot(1,1,1)
        ax1.plot(date, closep)
        ax1.grid(True)
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)
        plt.subplots_adjust(left=.10, bottom=.19, right=.93, top=.95, wspace=.20, hspace=.07)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.suptitle(stock+' Stock Price')
        #plt.show()
        plt.savefig('Graphs/'+stock+'.png', bbox_inches='tight')
    except Exception, e:
        print 'failed main loop',str(e)

def graphStocks(fileName):
    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            temp = row['ticker']
            pullData(temp)
            graphData(temp)

fileWithTickers = 'shortTickers.csv'
graphStocks(fileWithTickers)
