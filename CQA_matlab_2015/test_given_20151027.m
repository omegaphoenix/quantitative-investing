%%
clear all
load universe.mat
load given.mat
[~,i1,i2] = intersect(given.ticker,universe.stock.ticker);
given.ticker = given.ticker(i1);
given.beta = given.beta(i1);
given.quality = given.quality(i1);
universe.stock.ticker = universe.stock.ticker(i2);
universe.stock.fs = universe.stock.fs(i2);
universe.stock.ret = universe.stock.ret(i2);
universe.stock.week_ret = universe.stock.week_ret(i2);
universe.given = given;
%%
run('~/.matlab/cvx/cvx_startup.m')
%%
portfolio_start = '02-Jul-2012';
check_day = datewrkdy(portfolio_start,1:20*40);
balance_day = check_day(1:20:end);
%%
beta0 = calc_beta(universe.market.ret{1}, universe.stock.ret,datenum(portfolio_start),1/400);
%% test quality, alllong
[~,tmp]=sort(universe.given.quality);
selected_long_short_alllong = tmp;
saved_portfolio_alllong = backtest_portfolio(universe, selected_long_short_alllong, balance_day, check_day, @strategy_alllong);
%% test quality rank, half half
[~,tmp]=sort(universe.given.quality);
selected_long_short_half_half = tmp(end:-1:1);
saved_portfolio_half_half = backtest_portfolio(universe, selected_long_short_half_half, balance_day, check_day, @strategy_half_long_half_short);
%% test quality rank weighted strategy
[~,tmp]=sort(universe.given.quality);
selected_long_short_rankweighted = tmp(end:-1:1);
saved_portfolio_rankweighted = backtest_portfolio(universe, selected_long_short_rankweighted, balance_day, check_day, @strategy_rank_weight);
%% constrained by given beta, choose quality
[~,tmp]=sort(universe.given.quality);
selected_long_short_longshort50 = [tmp((1:50));tmp((end-49:end))];
saved_portfolio_longshort50 = backtest_portfolio(universe, selected_long_short_longshort50, balance_day, check_day, @bab_strategy_optimize3);
%% constrained by given beta, choose 100
[~,tmp]=sort(beta0);
selected_long_short_longshort100 = [tmp(1:100);tmp(end-99:end)];
saved_portfolio_longshort100 = backtest_portfolio(universe, selected_long_short_longshort100, balance_day, check_day, @bab_strategy_optimize2);
%% constrained by given beta, choose quality, 4
[~,tmp]=sort(universe.given.quality);
selected_long_short_longshort50_op4 = [tmp((1:50));tmp((end-49:end))];
saved_portfolio_longshort50_op4 = backtest_portfolio(universe, selected_long_short_longshort50, balance_day, check_day, @bab_strategy_optimize4);
%%
aqr = read_aqr();
%%
mkt_int = ret2tick(aqr.mkt);
bab_int = ret2tick(aqr.bab(10000:end));
cap1 = fints(cellfun(@(x) x.date,saved_portfolio_alllong)',cellfun(@(x) x.capital,saved_portfolio_alllong)','alllong',1);
cap2 = fints(cellfun(@(x) x.date,saved_portfolio_half_half)',cellfun(@(x) x.capital,saved_portfolio_half_half)','halfhalf',1);
cap3 = fints(cellfun(@(x) x.date,saved_portfolio_longshort50)',cellfun(@(x) x.capital,saved_portfolio_longshort50)','longshort50Reverse',1);
cap4 = fints(cellfun(@(x) x.date,saved_portfolio_longshort50_op4)',cellfun(@(x) x.capital,saved_portfolio_longshort50_op4)','longshort50Reverseop4',1);
plot_price_norm_at({mkt_int,bab_int,cap1,cap2,cap3,cap4},'02-Jul-2012');
%%
print -depsc2 a1_1027.eps
%%
plot(cellfun(@(x) universe.given.beta(selected_long_short_alllong)'*x.weight, saved_portfolio_alllong)); hold all
plot(cellfun(@(x) universe.given.beta(selected_long_short_half_half)'*x.weight, saved_portfolio_half_half)); hold all
plot(cellfun(@(x) universe.given.beta(selected_long_short_longshort50)'*x.weight, saved_portfolio_longshort50)); hold all
plot(cellfun(@(x) universe.given.beta(selected_long_short_longshort50_op4)'*x.weight, saved_portfolio_longshort50_op4)); hold all
legend alllong halfhalf longshort50Reverse longshort50Reverseop4
hold off
%%
print -depsc2 a2_1027.eps
