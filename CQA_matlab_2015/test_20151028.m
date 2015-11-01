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
aqr = read_aqr();
%%
portfolio_start = '02-Jul-2013';
check_day = datewrkdy(portfolio_start,1:20*90);
check_day = check_day(check_day <= datenum('Oct-18-2015'));
balance_day = check_day(1:20:end);
%%
beta0 = calc_beta(universe.market.ret{1}, universe.stock.ret,datenum(portfolio_start),1/400); beta0 = beta0*0.6+0.4;
beta1 = calc_beta(universe.market.ret{1}, universe.stock.ret,datenum('Oct-18-2015'),1/400);   beta1 = beta1*0.6+0.4;
%% Strats 1
[~,tmp]=sort(universe.given.quality);
selected_LS100 = [tmp(end:-1:end-99);tmp(100:-1:1)];
saved_portfolio_LS100 = backtest_portfolio(universe, selected_LS100, balance_day, check_day, @strategy_LS100);
%% Strats 2
[~,tmp]=sort(universe.given.quality);
selected_LS100_2 = flipud([tmp(end:-1:end-99);tmp(100:-1:1)]);
saved_portfolio_LS100_2 = backtest_portfolio(universe, selected_LS100_2, balance_day, check_day, @strategy_LS100);
%% Strats 3
[~,tmp]=sort(universe.given.quality);
selected_LS100_3 = [tmp(end:-1:end-99);tmp(100:-1:1)];
saved_portfolio_LS100_3 = backtest_portfolio(universe, selected_LS100_3, balance_day, check_day, @strategy_LS100_3);
%%
mkt_int = ret2tick(aqr.mkt);
bab_int = ret2tick(aqr.bab(10000:end));
cap1 = fints(cellfun(@(x) x.date,saved_portfolio_LS100)',cellfun(@(x) x.capital,saved_portfolio_LS100)','cap1',1);
cap2 = fints(cellfun(@(x) x.date,saved_portfolio_LS100_2)',cellfun(@(x) x.capital,saved_portfolio_LS100_2)','cap2',1);
cap3 = fints(cellfun(@(x) x.date,saved_portfolio_LS100_3)',cellfun(@(x) x.capital,saved_portfolio_LS100_2)','cap3',1);
plot_price_norm_at({mkt_int,cap1,cap2,cap3},'02-Jul-2013');
%%
hold all;
plot(cellfun(@(x) universe.given.beta(selected_LS100)'*x.weight, saved_portfolio_LS100));
plot(cellfun(@(x) universe.given.beta(selected_LS100_2)'*x.weight, saved_portfolio_LS100_2));
legend cap1 cap2 cap3
hold off