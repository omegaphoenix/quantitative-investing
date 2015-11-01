function weight = bab_strategy_reproduce_factor(universe, selected_ticker, check_day)

beta = calc_beta(universe.market.ret{1}, universe.stock.ret(selected_ticker), check_day,1/400);
beta = beta*0.6+0.4;
[~,~,ranking] = unique(beta);
weight = mean(ranking) - ranking;

%weight = sign(weight).*abs(weight).^(2);
weight = 2/sum(abs(weight))*weight;

long = weight>=0;
beta_long = beta(long)'*weight(long);
beta_short = beta(~long)'*weight(~long);
weight(long)=weight(long)/abs(beta_long);
weight(~long)=weight(~long)/abs(beta_short);