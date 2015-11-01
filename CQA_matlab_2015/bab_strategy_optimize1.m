function weight = bab_strategy_optimize1(universe, selected_ticker, check_day)

N = length(selected_ticker);
assert(N==100)

mu=zeros(N,1);
for i=1:N
    fs = universe.stock.ret{selected_ticker(i)};
    fs = fts2mat(fs);
    mu(i)=mean(fs(end-250:end));
end

[sigma,~,~] = calc_cov(universe,selected_ticker,datenum('2007-01-01'),check_day);
nb2 = N/2;


opt_mean=[];
opt_var=[];

%for lambda = logspace(-1,3,20)
for lambda = 1 
    cvx_begin
    variable  wL(nb2) nonnegative
    variable  wS(nb2) nonnegative
    w = [wL;-wS];
    maximize (mu'*w - lambda*(w'*sigma*w))
    subject to
    sum(wL) == 1
    sum(wS) == 1
    max(wL) <= 0.04
    max(wS) <= 0.04
    cvx_end
    
    opt_mean = [opt_mean; mu'*w];
    opt_var  = [opt_var ; w'*sigma*w];    
    
end

weight = w;




function weight = bab_strategy_optimize1_worker(universe, selected_ticker, check_day)

beta = calc_beta(universe.market.ret{1}, universe.stock.ret(selected_ticker), check_day,1/400);
beta = beta*0.6+0.4;



sigma = calc_cov(universe,selected_ticker,datenum('2007-01-01'),check_day);

n = length(beta);
lambda = 1000;

assert(n==100)

nb2 = n/2;

cvx_begin
    variable  wL(nb2) nonnegative
    variable  wS(nb2) nonnegative
    w = [wL;-wS];
    maximize (beta'*w - lambda*(w'*sigma*w))
    subject to
        sum(wL) == 1
        sum(wS) == 1
        max(wL) <= 0.04
        max(wS) <= 0.04    
cvx_end

weight = w;