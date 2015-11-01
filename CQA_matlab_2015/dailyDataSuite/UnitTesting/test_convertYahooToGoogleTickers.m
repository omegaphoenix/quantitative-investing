function tests = test_convertYahooToGoogleTickers
tests = functiontests(localfunctions);
end

% convertGoogleToYahooTickers converts Yahoo! Finance-formatted ticker
% symbols (TT.XX or just TT format, where XX is an exchange name and TT the
% ticker symbol itself) to Google Finance formatting (either TT.XX or just
% TT, depending on the exchange).  Here is its expected behaviour:

% An empty string input should return an empty string output rather than
% erroring.
function testEmptyString(testCase)
actSolution = convertYahooToGoogleTickers( '' );
expSolution = '';
verifyEqual(testCase,actSolution,expSolution);
end

% A single character string input ought to retun the appropriate output,
% also as a character string.
function testSingleAsset(testCase)
actSolution = convertYahooToGoogleTickers('ML.PA');
expSolution = 'EPA:ML';
verifyEqual(testCase,actSolution,expSolution);
end

% US stocks have no exchange info in Yahoo!.  In Google, they might have
% NYSE: or NASDAQ:. Because we can't determine which, just return the
% ticker itself and hope that Google can resolve it from there.
function testSingleTickerOnlyAsset(testCase)
actSolution = convertYahooToGoogleTickers('MSFT');
expSolution = 'MSFT';
verifyEqual(testCase,actSolution,expSolution);
end

% Multiple tickers can be input at once as a cell array of strings.  The
% above rules apply on a cell-by-cell basis, and we're agnostic as to the
% shape of the input cell array. Either a row vector:
function testRowOfAssets(testCase)
actSolution = convertYahooToGoogleTickers({'MSFT', 'ML.PA', ''});
expSolution = {'MSFT', 'EPA:ML', ''};
verifyEqual(testCase,actSolution,expSolution);
end

% ... or a column vector.
function testColumnOfAssets(testCase)
actSolution = convertYahooToGoogleTickers({'MSFT'; 'ML.PA'; ''});
expSolution = {'MSFT'; 'EPA:ML'; ''};
verifyEqual(testCase,actSolution,expSolution);
end

% Remove warning about unused local functions, since |localfunctions| does
% use all of them:
%#ok<*DEFNU>