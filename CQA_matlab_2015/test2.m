clear all
load universe.mat
%%
run('~/.matlab/cvx/cvx_startup.m')
%%
portfolio_start = '02-Jul-2012';
check_day = datewrkdy(portfolio_start,1:20*40);
balance_day = check_day(1:20:end);
%%
saved_portfolio = backtest_portfolio(universe, 1:1000, balance_day, check_day, @bab_strategy_reproduce_factor);
%%
saved_portfolio = backtest_portfolio(universe, 1:length(universe.stock.ticker), check_day(1), check_day, @bab_strategy_longshort50);
%%
beta0 = calc_beta(universe.market.ret{1}, universe.stock.ret,datenum(portfolio_start),1/400);
%% seems very good backtest
[~,tmp]=sort(beta0);
selected_long_short = [tmp(1:50);tmp(end-49:end)];
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