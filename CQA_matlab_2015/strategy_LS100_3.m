function weight = strategy_LS100_3(universe, selected_ticker, check_day)

N = length(selected_ticker);
assert(N==200);

beta_raw = calc_beta(universe.market.week_ret{1}, universe.stock.week_ret(selected_ticker), check_day,1/400);
beta = beta_raw*0.6+0.4;

[sigma,~,~] = calc_cov(universe,selected_ticker,datenum('2007-01-01'),check_day);

eig_min = eigs(sigma,1,'sa');
if eig_min < 1E-5
    sigma = sigma + max(2E-5,-2*eig_min)*eye(N);
end
nb2 = N/2;


given_beta = universe.given.beta(selected_ticker);

%for lambda = logspace(-1,3,20)
for lambda = 10 
    cvx_begin
    
    variable  wL(nb2) nonnegative
    variable  wS(nb2) nonnegative
    w = [wL;-wS];
    minimize (beta'*w + lambda*(w'*sigma*w))
    subject to
    sum(wL) == 1
    sum(wS) == 1
    max(wL) <= 0.02
    max(wS) <= 0.02
    given_beta'*w <= 0.4
    given_beta'*w >= -0.4
    
    cvx_end 
    
end

weight = w;