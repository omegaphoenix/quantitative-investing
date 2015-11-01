function weight = strategy_half_long_half_short(universe, selected_ticker, check_day)

n=floor(length(selected_ticker)/2);
weight = ones(length(selected_ticker),1);
weight(n:end)=-1;
weight = 2/sum(abs(weight))*weight;
