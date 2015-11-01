function plot_price_norm_at(prices, start)

for i=1:length(prices)
    prices{i} = prices{i}/fts2mat(prices{i}(start));
end

A = merge(prices{:},'DateSetMethod','Intersection');
A = A(A.dates >= datenum(start));
plot(A);
    
    