function saved_portfolio = backtest_portfolio(universe, selected_ticker, balance_day, check_day, func_newweight)

Nticker = length(selected_ticker);

% initial holding
holding.bank = 1E7;
holding.stock_amount = zeros(Nticker,1);
holding.capital = holding.bank;

saved_portfolio = {};

check_day = sort(union(balance_day, check_day));

lastest_price = get_lastest_price(universe.stock, selected_ticker, check_day);
next_price    = get_next_price(universe.stock, selected_ticker, check_day);

for i=1:length(check_day)
    
    holding.capital = holding.bank + lastest_price(i,:)*holding.stock_amount;
    holding.date = check_day(i);
    
    
    holding.transaction.amount = zeros(Nticker,1);
    holding.transaction.price  = zeros(Nticker,1);
    
    if( ~isempty(find(balance_day == check_day(i))) )
        
        % call back function to calculate new weight
        weight = func_newweight(universe, selected_ticker, check_day(i));  
        
        % rebalance in the next trading day
        amount_to_buy = weight*holding.capital./lastest_price(i,:)' - holding.stock_amount;
        dolar_to_pay = next_price(i,:)*amount_to_buy;
        
        
        holding.transaction.amount = amount_to_buy;
        holding.transaction.price  = next_price(i,:)';

        holding.bank = holding.bank - dolar_to_pay;
        holding.stock_amount = holding.stock_amount + amount_to_buy;
    end
    
    holding.weight = (lastest_price(i,:)'.*holding.stock_amount)/holding.capital;
    saved_portfolio{i} = holding;
    long = weight > 0;
    fprintf('%d: %s %f %f %f %f %f\n',i, datestr(check_day(i)), holding.capital, sum(weight(long)), sum(weight(~long)), max(weight(long)), min(weight(~long)));
end

    
