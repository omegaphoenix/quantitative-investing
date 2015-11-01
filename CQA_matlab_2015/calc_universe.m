function calc_universe()
load Yahoo_20151030_r3000.mat
start_date=[]; end_date=[]; total_day=[];
for i=1:length(stock.data)
    start_date(i) = stock.data{i}.Date(1);
    end_date(i) = stock.data{i}.Date(end);
    total_day = length(stock.data{i}.Date);
end

per=(end_date >= datenum('2015-10-16') & start_date <= datenum('2010-01-01'));

stock.data = stock.data(per);
stock.ticker = stock.ticker(per);


tmp={};
for i=1:length(stock.data)
    tmp{i} = fints(stock.data{i}.Date, stock.data{i}.AdjClose,'AdjClose',1);
end
stock.fs=tmp;
tmp={};
for i=1:length(market.data)
    tmp{i} = fints(market.data{i}.Date, market.data{i}.AdjClose,'AdjClose',1);
end
market.fs=tmp;
universe.stock = rmfield(stock,'data');
universe.market = rmfield(market,'data');

universe.stock.ret = cellfun(@(x) tick2ret(x,'method','continuous'), universe.stock.fs,'UniformOutput',false);
universe.market.ret = cellfun(@(x) tick2ret(x,'method','continuous'), universe.market.fs,'UniformOutput',false);
universe.stock.week_ret = cellfun(@(x) toweekly(x,'calcmethod','cumsum'), universe.stock.ret,'UniformOutput',false);
universe.market.week_ret = cellfun(@(x) toweekly(x,'calcmethod','cumsum'), universe.market.ret,'UniformOutput',false);

save universe_1030.mat universe
