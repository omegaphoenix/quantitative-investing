function tests = test_convertGoogleToYahooTickers
tests = functiontests(localfunctions);
end

% convertGoogleToYahooTickers converts Google Finance-formatted ticker
% symbols (XX:TT format, where XX is an exchange name and TT the ticker
% symbol itself) to Yahoo! Finance formatting (either TT.XX or just TT,
% depending on the exchange).  Here is its expected behaviour:

% An empty string input should return an empty string output rather than
% erroring.
function testEmptyString(testCase)
actSolution = convertGoogleToYahooTickers( '' );
expSolution = '';
verifyEqual(testCase,actSolution,expSolution);
end

% A single character string input ought to retun the appropriate output,
% also as a character string.
function testSingleAsset(testCase)
actSolution = convertGoogleToYahooTickers('EPA:ML');
expSolution = 'ML.PA';
verifyEqual(testCase,actSolution,expSolution);
end

% An exchange that is unrecognized by Google or Yahoo should be returned as
% an empty string rather than erroring.
function testSingleGoogleOnlyAsset(testCase)
% Only Google tracks the Tokyo stock exchange
actSolution = convertGoogleToYahooTickers('TYO:7974');
expSolution = '';
verifyEqual(testCase,actSolution,expSolution);
end

% Multiple tickers can be input at once as a cell array of strings.  The
% above rules apply on a cell-by-cell basis, and we're agnostic as to the
% shape of the input cell array.
% Either a row vector:
function testRowOfAssets(testCase)
actSolution = convertGoogleToYahooTickers({'NASDAQ:MSFT', 'EPA:ML', 'TYO:7974'});
expSolution = {'MSFT', 'ML.PA', ''};
verifyEqual(testCase,actSolution,expSolution);
end

% ... or a column vector.
function testColumnOfAssets(testCase)
actSolution = convertGoogleToYahooTickers({'NASDAQ:MSFT'; 'EPA:ML'; 'TYO:7974'});
expSolution = {'MSFT'; 'ML.PA'; ''};
verifyEqual(testCase,actSolution,expSolution);
end

% Remove warning about unused local functions, since |localfunctions| does
% use all of them:
%#ok<*DEFNU>