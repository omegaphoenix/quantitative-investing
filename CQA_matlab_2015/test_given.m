% 20151026
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
selected_long_short = 1:length(universe.stock.ticker);
saved_portfolio = backtest_portfolio(universe,selected_long_short  , balance_day, check_day, @bab_strategy_reproduce_factor);
%%
saved_portfolio = backtest_portfolio(universe, 1:length(universe.stock.ticker), check_day(1), check_day, @bab_strategy_longshort50);
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
%% constrained by given beta, choose quality, 4
[~,tmp]=sort(universe.given.quality);
selected_long_short_longshort50_op4 = [tmp((1:50));tmp((end-49:end))];
saved_portfolio_longshort50_op4 = backtest_portfolio(universe, selected_long_short_longshort50, balance_day, check_day, @bab_strategy_optimize4);

%% constrained by given beta
[~,tmp]=sort(beta0);
selected_long_short_longshort100 = [tmp(1:100);tmp(end-99:end)];
saved_portfolio_longshort100 = backtest_portfolio(universe, selected_long_short_longshort100, balance_day, check_day, @bab_strategy_optimize2);
%% seems very good backtest
[~,tmp]=sort(beta0);
selected_long_short = [tmp(1:50);tmp(end-49:end)];
selected_long_short = selected_long_short(end:-1:1);
saved_portfolio = backtest_portfolio(universe, selected_long_short, balance_day, check_day, @bab_strategy_optimize1);
%%
trans_amount = cellfun(@(x) x.transaction.amount, saved_portfolio,'uniformoutput',false);
trans_price = cellfun(@(x) x.transaction.price, saved_portfolio,'uniformoutput',false);
trans_amount = cell2mat(trans_amount);
trans_price = cell2mat(trans_price);
tmp = find(trans_price(1,:)~=0);
trans_amount=trans_amount(:,tmp);
trans_price =trans_price(:,tmp);

paid = trans_amount*0;
worth= trans_amount*0;
for i=1:size(trans_amount,1)
    for t=1:size(trans_amount,2)
        paid(i,t) = sum(trans_amount(i,1:t).*trans_price(i,1:t));
        worth(i,t) = sum(trans_amount(i,1:t))*trans_price(i,t);    
    end    
end

%%
[~,tmp]=sort(beta0);
selected_long_short = [tmp(1:50);tmp(end-49:end)];
saved_portfolio = backtest_portfolio(universe, 1:100, balance_day, check_day, @bab_strategy_optimize1);
%%
aqr = read_aqr();
%%
mkt_int = ret2tick(aqr.mkt);
bab_int = ret2tick(aqr.bab(10000:end));
cap_fs = fints(cellfun(@(x) x.date,saved_portfolio)',cellfun(@(x) x.capital,saved_portfolio)','cap',1);
plot_price_norm_at({mkt_int,bab_int,cap_fs},'02-Jul-2012');
%%
plot(cellfun(@(x) beta0(selected_long_short)'*x.weight, saved_portfolio))
%%
plot(cellfun(@(x) universe.given.beta(selected_long_short)'*x.weight, saved_portfolio))