function tests = test_getGoogleDailyData
tests = functiontests(localfunctions);
end

% getGoogleDailgData gets the OHLC, volume, and adjusted closing prices for
% specified assets from Google Finance.

% Invalid or blank tickers should return an error.
function testBadTicker(testCase)
fcn = @()getGoogleDailyData('QuantSupport', '01/01/2010', '01/01/2013', 'dd/mm/yyyy');
verifyError(testCase,fcn,'getGoogleDailyData:invalidTicker');
end

% NOTE: Financial data is always subject to revision.  There is no
% guarantee that Google will return the same answers every time we run this
% function (and in fact we all but guarantee that the adjusted closing
% prices will vary).  For this reason, failure of any part of the following
% tests is not a guarantee that the function is incorrect.  We recommend
% running these tests BEFORE making any code changes and re-saving the
% expected solutions if needed.

% Function can have three inputs if startDate and endDate are datenums...
function testValidDataWithDatenums(testCase)
actSolution = getGoogleDailyData('NASDAQ:MSFT', 734139, 735235);
expSolution = load('Data_test_DailyData', 'gData');
verifyEqual(testCase, actSolution.NASDAQ0x3AMSFT{:,1:6}, expSolution.gData.NASDAQ0x3AMSFT{:,1:6});
end

% Function can have 4 inputs if the date format is given.
function testValidDataWithDatestrings(testCase)
actSolution = getGoogleDailyData('NASDAQ:MSFT', '01/01/2010', '01/01/2013', 'dd/mm/yyyy');
expSolution = load('Data_test_DailyData', 'gData');
verifyEqual(testCase, actSolution.NASDAQ0x3AMSFT{:,1:6}, expSolution.gData.NASDAQ0x3AMSFT{:,1:6});
end

% Function also accepts a cell array of 1 or more tickers.
function testValidDataWithMultipleTickers(testCase)
gTickers = load('Data_test_DailyData', 'gTickers');
gTickers = gTickers.gTickers;
expSolution = load('Data_test_DailyData', 'gData');

actSolution = getGoogleDailyData(gTickers, '01/01/2010', '01/01/2013', 'dd/mm/yyyy');

for iTicker = 1:length(gTickers)
    verifyEqual(testCase, actSolution.(genvarname(gTickers{iTicker})){:,1:6}, ...
        expSolution.gData.(genvarname(gTickers{iTicker})){:,1:6});
end
end

% Function gives ex-dividend dates and dividends as the second output
function testValidDataDividends(testCase)
gTickers = load('Data_test_DailyData', 'gTickers');
gTickers = gTickers.gTickers;
expSolution = load('Data_test_DailyData', 'dividends');

[~,actSolution] = getGoogleDailyData(gTickers, '01/01/2010', '01/01/2013', 'dd/mm/yyyy');

for iTicker = 1:length(gTickers)
    verifyEqual(testCase, actSolution.(genvarname(gTickers{iTicker})){:,:}, ...
        expSolution.dividends.(genvarname(gTickers{iTicker})){:,:});
end
end

% Remove warning about unused local functions, since |localfunctions| does
% use all of them:
%#ok<*DEFNU>