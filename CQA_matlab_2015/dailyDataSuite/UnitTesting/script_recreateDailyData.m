%% Script to re-create Daily Data for tests
% Financial data is always subject to revision.  There is no guarantee that
% Yahoo! and Google will return the same answers every time we run their
% daily data functions (and in fact we all but guarantee that the adjusted
% closing prices will vary).  For this reason, failure of their unit tests
% is not a guarantee that anything is wrong with the functions themselves.
%
% We recommend that you run the unit tests test_getGoogleDailyData and
% test_getYahooDailyData _before_ making any changes to those files.  If
% they fail, determine if the test data set has changed.  If so, then run
% this script to update the "correct" data-- be careful that you trust the
% version of your data loaders when running this script!

%% Specify valid Google and Yahoo! tickers
% You may specify any valid ticker symbols here.  The script will later add
% Microsoft (if it isn't already there) since it is hard-coded into some of
% the single-asset tests.

yTickers = {'MSFT', 'ML.PA'};
gTickers = {'NASDAQ:MSFT', 'EPA.ML'};

%% Grab and save data with the appropriate date range:
yTickers = union(yTickers, 'MSFT');
gTickers = union(gTickers, 'NASDAQ:MSFT');
yData = getYahooDailyData(yTickers, ...
    '01/01/2010', '01/01/2013', 'dd/mm/yyyy');
[gData, dividends] = getGoogleDailyData(gTickers, ...
    '01/01/2010', '01/01/2013', 'dd/mm/yyyy'); 

save('Data_test_DailyData', 'yTickers', 'gTickers', 'yData', 'gData', 'dividends');

clear yTickers gTickers yData gData dividends

% Remove code analyzer warnings:
%#ok<*NASGU>
%#ok<*ASGLU>