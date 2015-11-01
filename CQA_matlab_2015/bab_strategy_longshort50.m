function weight = bab_strategy_longshort50(universe, selected_ticker, check_day)

beta = calc_beta(universe.market.ret{1}, universe.stock.ret, check_day,1/400);
beta = beta*0.6+0.4;
[~,~,ranking] = unique(beta);

weight = ranking*0;
%weight(ranking<=50) = 1.0;
weight(ranking>=length(ranking)-50) = -1.0;
%weight = weight + 1.0;


weight = 1/sum(abs(weight))*weight;

long = weight>=0;
beta_long = beta(long)'*weight(long);
beta_short = beta(~long)'*weight(~long);
weight(long)=weight(long)/(1E-10+abs(beta_long));
weight(~long)=weight(~long)/(1E-10+abs(beta_short));

weight = weight/sum(abs(weight));

1;