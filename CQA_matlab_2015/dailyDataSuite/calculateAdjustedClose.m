function adjClose = calculateAdjustedClose(closeDates, closePrices, dividendDates, dividendPrices)
% Applies the CRSP method for adjusting closing prices for dividends.
% Each new dividend changes the adjusted close for all dates in the past,
% so strictly speaking this only calculates the adjusted closing prices
% as measured on max(closeDates).
%
% closeDates: an N-by-1 vector of datenums for the closing price series.
% closePrices: an N-by-1 vector of closing prices.
% (The above two MUST be sorted so that the most recent price is in row 1!)
%
% dividendDates: an M-by-1 vector of datenums for the dividends
% dividendPrices: an M-by-1 vector of dividend amounts in units of currency

% Ensure that closeDates is in descending order-- this is crucial to get
% right due to the |cumprod| below and will not otherwise be caught:
if ~all(diff(closeDates) < 0)
    error('calculateAdjustedClose:CloseSorted', ...
        'Closing prices/dates must be sorted in decreasing order')
end

if isempty(closeDates)
    % Empty closing prices ought to lead to empty adjusted closing prices -
    % not an error.
    adjClose = closePrices;
    return
end

adjustmentFactor = ones(size(closePrices,1),1);
for iDiv = 1:size(dividendPrices,1)
    % Since the dividend is paid overnight going INTO the
    % ex-dividend date, we start to apply the adjustment to the
    % closing price on the day BEFORE the ex-dividend date.
    
    % Also, the list occasionally has copies of dividends.  We must make
    % sure that we don't double-count these.
    idx = find(closeDates < dividendDates(iDiv), 1);
    adjustmentFactor(idx) = 1 - (dividendPrices(iDiv)/closePrices(idx));
end

% Accumulate the adjustment factors and multiply them against the close
adjClose = cumprod(adjustmentFactor) .* closePrices;
% Round to the nearest cent
adjClose = round(100*adjClose)/100;