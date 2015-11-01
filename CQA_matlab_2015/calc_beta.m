function beta = calculate_beta(market,stock,end_date,decay)
beta =[];
fs1= fts2mat(market,1);
for i=1:length(stock)    
    fs2 = fts2mat(stock{i},1); 
    beta(i) = calc_beta_one(fs1,fs2,end_date-365*8, end_date,decay);    
end
beta = beta';


function beta = calc_beta_one(fs1,fs2,start_date,end_date,decay)

% half life = 1/decay

fs1 = fs1(fs1(:,1) >= start_date & fs1(:,1) <= end_date,:);
fs2 = fs2(fs2(:,1) >= start_date & fs2(:,1) <= end_date,:);


[~,i1,i2] = intersect(fs1(:,1), fs2(:,1));

date = fs1(i1,1);
weight = exp((date-date(end))*decay);

ret1 = fs1(i1,2);
ret2 = fs2(i2,2);

ret1(isnan(ret1))=0;
ret2(isnan(ret2))=0;

%lm = fitlm(ret1,ret2,'weight',weight);
%beta = lm.Coefficients.Estimate(2);

X = [ret1*0+1,ret1];
XtW = [X(:,1).*weight,X(:,2).*weight]';
lm = (XtW*X)\(XtW*ret2);
beta = lm(2);

% std1 = std(ret1(max(1,end-250):end));
% std2 = std(ret2(max(1,end-250):end));
% 
% sm1 = smooth(ret1(max(1,end-250*5):end),3);
% sm2 = smooth(ret2(max(1,end-250*5):end),3);
% 
% rho  = corr(sm1(1:3:end),sm2(1:3:end));
% beta = rho*std2/std1;
1;