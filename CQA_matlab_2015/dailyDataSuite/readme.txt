This archive contains functions for downloading daily stock price information from both Google and Yahoo! Finance as well as helpful utility functions.  It also contains a (basic but decent) example of a unit test harness created using MATLAB's functional unit test framework.

The price downloading commands and the unit test functions require features that appeared in MATLAB release 2013b.  If you are using an older version of MATLAB, then the unit tests will not work.  The rest of the functionality will work, but the price downloading commands will require Statistics Toolbox and will return dataset arrays instead of tables.

Both Google and Yahoo! seem to allow the use of financial data for PERSONAL, INFORMATIONAL use ONLY and do not warrant against inaccuracies.  These discrepancies are fairly common and are the main reason that this suite uses both Yahoo! AND Google data-- only if both sources agree on a price of data should you begin to trust it.

For questions and discussion, go to www.quantsupport.com.

List of primary functions:

getGoogleDailyData: Downloads OHLC and volume info for one or more tickers from Google for a given range of dates.  Also downloads dividend information and uses that to calculate adjusted closing prices.

getYahooDailyData: Download OHLC, volume, and adjusted closing price info for one or more tickers from Yahoo! for a given range of dates.

convertGoogleToYahooTickers: Converts ticker symbols formatted for Google finance to those that are recognised by Yahoo! finance.  (This is crucial for non-US stocks and can be important even for US stocks.)

convertGoogleToYahooTickers: The reverse of the above.

calculateAdjustedClose: Uses raw closing price and dividend information to calculate the adjusted closing price.  It called automatically within getGoogleDailyData so that it returns the same fields as getYahooDailyData.

Unit testing functions:
These are all the files in the UnitTesting folder as well as the following:

runAllTests: A helper script that sets the appropriate path information and then all unit test functions for this suite.  Look at it for the proper syntax of calling any given unit test individually.

convertGoogleToYahooTickers_BUG: A buggy version of convertGoogleToYahooTickers-- it contains a subtle bug (_very_ subtle, in fact) and as a result will fail the test_convertGoogleToYahooTickers unit test.  To make it the active file so you can see what a failed unit test looks like, change the name of convertGoogleToYahooTickers to something else and change its name to convertGoogleToYahooTickers.  Don't forget to change the names back when you're done!

UnitTesting/script_recreateDailyData: It's the nature of financial data to be subject to revision, so we would expect some of the unit root tests to fail afer a while simply because either Yahoo! or Google changed the historical record.  If you are certain that this is the reason the tests are failing (and not because of a bug in the code), then you can run this function to overwrite the stored "expected" answers used by test_getGoogleDailyData and test_getYahooDailyData.  Use with caution!