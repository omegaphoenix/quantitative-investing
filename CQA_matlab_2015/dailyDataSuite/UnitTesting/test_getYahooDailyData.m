function tests = test_getYahooDailyData
tests = functiontests(localfunctions);
end

% getYahooDailyData gets the OHLC, volume, and adjusted closing prices for
% specified assets from Yahoo! Finance.

% Invalid or blank tickers should return an error.
function testBadTicker(testCase)
fcn = @()getYahooDailyData('QuantSupport', '01/01/2010', '01/01/2013', 'dd/mm/yyyy');
verifyError(testCase,fcn,'getYahooDailyData:invalidTicker');
end

% NOTE: Financial data is always subject to revision.  There is no
% guarantee that Yahoo! will return the same answers every time we run this
% function (and in fact we all but guarantee that the adjusted closing
% prices will vary).  For this reason, failure of any part of the following
% tests is not a guarantee that the function is incorrect.  We recommend
% running these tests BEFORE making any code changes and re-saving the
% expected solutions if needed.

% Function can have three inputs if startDate and endDate are datenums...
function testValidDataWithDatenums(testCase)
actSolution = getYahooDailyData('MSFT', 734139, 735235);
expSolution = load('Data_test_DailyData', 'yData');
verifyEqual(testCase, actSolution.MSFT{:,1:6}, expSolution.yData.MSFT{:,1:6});
end

% Function can have 4 inputs if the date format is given.
function testValidDataWithDatestrings(testCase)
actSolution = getYahooDailyData('MSFT', '01/01/2010', '01/01/2013', 'dd/mm/yyyy');
expSolution = load('Data_test_DailyData', 'yData');
verifyEqual(testCase, actSolution.MSFT{:,1:6}, expSolution.yData.MSFT{:,1:6});
end

% Function also accepts a cell array of 1 or more tickers.
function testValidDataWithMultipleTickers(testCase)
yTickers = load('Data_test_DailyData', 'yTickers');
yTickers = yTickers.yTickers;
expSolution = load('Data_test_DailyData', 'yData');

actSolution = getYahooDailyData(yTickers, '01/01/2010', '01/01/2013', 'dd/mm/yyyy');

for iTicker = 1:length(yTickers)
    verifyEqual(testCase, actSolution.(genvarname(yTickers{iTicker})){:,1:6}, ...
        expSolution.yData.(genvarname(yTickers{iTicker})){:,1:6});
end
end

% Remove warning about unused local functions, since |localfunctions| does
% use all of them:
%#ok<*DEFNU>