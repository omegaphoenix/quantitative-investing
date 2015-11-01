function tests = test_calculateAdjustedClose
tests = functiontests(localfunctions);
end

% calculateAdjustedClose finds the adjusted closing prices given the actual
% closing prices/dates, the ex-dividend dates, and dividend amounts.  It
% follows the method used by Yahoo! Finance and originally described by the
% CRSP.  Here is its expected behaviour:

% Empty closing price inputs should return an empty output rather than
% erroring.
function testEmptyClose(testCase)
actSolution = calculateAdjustedClose([], [], [], []);
expSolution = [];
verifyEqual(testCase,actSolution,expSolution);
end

% The closing dates should be in descending chronological order-- anything
% else should error.
function testAscendingCloseDates(testCase)
fcn = @()calculateAdjustedClose((today:today+9)', rand(10,1), [], []);
verifyError(testCase,fcn,'calculateAdjustedClose:CloseSorted')
end

% If there are no dividends (empty arrays), then the adjusted close must be
% the exact same as the close:
function testNoDividends(testCase)
data = load('Data_test_calculateAdjustedClose');
actSolution = calculateAdjustedClose(data.closeDates, data.closePrices, [], []);
expSolution = data.closePrices;
verifyEqual(testCase,actSolution,expSolution);
end

% In a "typical" dividend situation, give the correct closing prices to
% within 0.01 (one cent / Euro cent / penny / etc.)
function testTypicalDividends(testCase)
data = load('Data_test_calculateAdjustedClose');
actSolution = calculateAdjustedClose(data.closeDates, data.closePrices, ...
    data.dividendDates, data.dividendPrices);
expSolution = data.adjClose;
verifyEqual(testCase,actSolution,expSolution, 'AbsTol', 0.01);
end

% Remove warning about unused local functions, since |localfunctions| does
% use all of them:
%#ok<*DEFNU>