function weight = strategy_alllong(universe, selected_ticker, check_day)

weight = ones(length(selected_ticker),1);
weight = 1/sum(weight)*weight;
