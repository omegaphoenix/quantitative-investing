function r = get_next_price(universe, selected_ticker, dates)
r = zeros(length(dates),length(selected_ticker));
for i=1:length(selected_ticker)
    tmp = universe.fs{selected_ticker(i)};    
    r(:,i) = interp1(tmp.dates, fts2mat(tmp), dates(:)+1,'next');
end
