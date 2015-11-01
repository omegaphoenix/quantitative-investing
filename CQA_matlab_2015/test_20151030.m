clear all
load universe_1030.mat
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
check_day = check_day(check_day <= datenum('Oct-30-2015'));
balance_day = check_day(1:20:end);
%%
beta0 = calc_beta(universe.market.ret{1}, universe.stock.ret,datenum(portfolio_start),1/400); beta0 = beta0*0.6+0.4;
beta1 = calc_beta(universe.market.ret{1}, universe.stock.ret,datenum('Oct-30-2015'),1/400);   beta1 = beta1*0.6+0.4;
%% execuation 
[~,tmp]=sort(universe.given.quality);
lastest_price_tmp = get_lastest_price(universe.stock, tmp, datenum('Oct-30-2015'));
tmp = tmp(lastest_price_tmp > 10); 
exec_LS200 = [tmp(end:-1:end-99);tmp(100:-1:1)];
exec_weight = strategy_LS100_3(universe, exec_LS200, datenum('Oct-30-2015'));
%%
holding_amount = zeros(length(exec_LS200),1);
lastest_price = get_lastest_price(universe.stock, exec_LS200, datenum('Oct-30-2015'));
capital = 1E7;
amount = capital.*exec_weight ./ lastest_price';
amount = round(amount/100)*100;
fid = fopen('exec_20151030.txt','w');

for i=1:length(exec_LS200)
    if (amount(i) > holding_amount(i) && holding_amount(i) >= 0 && amount(i) >= 0)  
        fprintf(fid,'%s %s %d\n',  'Buy', universe.stock.ticker{exec_LS200(i)}, amount(i)-holding_amount(i));
    end
    if (amount(i) > holding_amount(i) && holding_amount(i) >= 0 && amount(i) < 0)  
    end  
    if (amount(i) > holding_amount(i) && holding_amount(i) < 0 && amount(i) <= 0)
        fprintf(fid,'%s %s %d\n',  'Cover', universe.stock.ticker{exec_LS200(i)},amount(i)-holding_amount(i));
    end
    if (amount(i) > holding_amount(i) && holding_amount(i) < 0 && amount(i) > 0)
        fprintf(fid,'%s %s %d\n',  'Cover', universe.stock.ticker{exec_LS200(i)},-holding_amount(i));
        fprintf(fid,'%s %s %d\n',  'Buy', universe.stock.ticker{exec_LS200(i)},amount(i));
    end 
    
    if (amount(i) < holding_amount(i) && holding_amount(i) > 0 && amount(i) >= 0)  
        fprintf(fid,'%s %s %d\n',  'Sell', universe.stock.ticker{exec_LS200(i)}, -amount(i)+holding_amount(i));
    end
    if (amount(i) < holding_amount(i) && holding_amount(i) > 0 && amount(i) < 0)  
        fprintf(fid,'%s %s %d\n',  'Sell', universe.stock.ticker{exec_LS200(i)},holding_amount(i));
        fprintf(fid,'%s %s %d\n',  'Short', universe.stock.ticker{exec_LS200(i)},-amount(i));
    end  
    if (amount(i) < holding_amount(i) && holding_amount(i) <= 0 && amount(i) <= 0)
        fprintf(fid,'%s %s %d\n',  'Short', universe.stock.ticker{exec_LS200(i)},-amount(i)+holding_amount(i));
    end
    if (amount(i) < holding_amount(i) && holding_amount(i) <= 0 && amount(i) > 0)
    end 
           
end
fclose(fid);
%%


























%%% Strats 1
%[~,tmp]=sort(universe.given.quality);
%selected_LS100 = [tmp(end:-1:end-99);tmp(100:-1:1)];
%saved_portfolio_LS100 = backtest_portfolio(universe, selected_LS100, balance_day, check_day, @strategy_LS100);
%%% Strats 2
%[~,tmp]=sort(universe.given.quality);
%selected_LS100_2 = flipud([tmp(end:-1:end-99);tmp(100:-1:1)]);
%saved_portfolio_LS100_2 = backtest_portfolio(universe, selected_LS100_2, balance_day, check_day, @strategy_LS100);
%%% Strats 3
%[~,tmp]=sort(universe.given.quality);
%selected_LS100_3 = [tmp(end:-1:end-99);tmp(100:-1:1)];
%saved_portfolio_LS100_3 = backtest_portfolio(universe, selected_LS100_3, balance_day, check_day, @strategy_LS100_3);
%%%
%mkt_int = ret2tick(aqr.mkt);
%bab_int = ret2tick(aqr.bab(10000:end));
%cap1 = fints(cellfun(@(x) x.date,saved_portfolio_LS100)',cellfun(@(x) x.capital,saved_portfolio_LS100)','cap1',1);
%cap2 = fints(cellfun(@(x) x.date,saved_portfolio_LS100_2)',cellfun(@(x) x.capital,saved_portfolio_LS100_2)','cap2',1);
%cap3 = fints(cellfun(@(x) x.date,saved_portfolio_LS100_3)',cellfun(@(x) x.capital,saved_portfolio_LS100_2)','cap3',1);
%plot_price_norm_at({mkt_int,cap1,cap2,cap3},'02-Jul-2013');
%%%
%hold all;
%plot(cellfun(@(x) universe.given.beta(selected_LS100)'*x.weight, saved_portfolio_LS100));
%plot(cellfun(@(x) universe.given.beta(selected_LS100_2)'*x.weight, saved_portfolio_LS100_2));
%legend cap1 cap2 cap3
%hold off
