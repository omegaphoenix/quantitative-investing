function weight = bab_strategy_optimize3(universe, selected_ticker, check_day)

N = length(selected_ticker);
assert(N==100)

mu=zeros(N,1);
for i=1:N
    fs = universe.stock.ret{selected_ticker(i)};
    fs = fts2mat(fs);
    mu(i)=mean(fs(end-250:end));
end

[sigma,~,~] = calc_cov(universe,selected_ticker,datenum('2007-01-01'),check_day);

eig_min = eigs(sigma,1,'sa');
if eig_min < 1E-5
    sigma = sigma + max(2E-5,-2*eig_min)*eye(N);
end
nb2 = N/2;


given_beta = universe.given.beta(selected_ticker);

opt_mean=[];
opt_var=[];

%for lambda = logspace(-1,3,20)
for lambda = 10 
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
    given_beta'*w <= 0.4
    given_beta'*w >= -0.4
    
    cvx_end
    
    opt_mean = [opt_mean; mu'*w];
    opt_var  = [opt_var ; w'*sigma*w];    
    
end

weight = w;