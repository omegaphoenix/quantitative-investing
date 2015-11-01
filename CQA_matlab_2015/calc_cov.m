function [C,lambda,S] = calc_cov(universe, selected_ticker, start_date, end_date)

N = length(selected_ticker);
% selection
fs = {};
for i=1:N
    tmp = fts2mat(universe.stock.ret{i},1);
    tmp = tmp(tmp(:,1) >= start_date & tmp(:,1) <= end_date,:);
    fs{i} = tmp;
end

S = zeros(N,N);
PI = S; Tiiij = S; Tjjij=S; T=S;



for i=1:N
    for j=1:N
        [~,i1,i2] = intersect(fs{i}(:,1), fs{j}(:,1));
        a1 = fs{i}(i1,2);
        a2 = fs{j}(i2,2);        
        [PI(i,j), S(i,j),Tiiij(i,j),Tjjij(i,j)] = calc(a1,a2);   
        T(i,j) = length(a1);
    end
end

V = sqrt(diag(S)); V = V*V';
R = S ./ V;

I = triu(true(N)); 
for i=1:N; I(i,i)=0; end
Rbar=mean(R(I));

F = Rbar*V;
for i=1:N
    F(i,i) = S(i,i);
end

rho = sum(diag(PI));
for i=1:N
    for j=1:N
        if(j~=i)
            rho = rho + Rbar/2*(sqrt(S(j,j)/S(i,i))*Tiiij(i,j)+sqrt(S(i,i)/S(j,j))*Tjjij(i,j)); 
        end
    end
end

spi = sum(PI(:));
gamma = sum(sum( (F-S).^2));

k = (spi-rho)/gamma;
lambda = max(0,min(k/mean(T(:)),1));
C = lambda*F + (1-lambda)*S;



function [p_ij, s_ij, t_iiij, t_jjij] = calc(ai,aj)
T = length(ai);
ai = ai-mean(ai);
aj = aj-mean(aj);
s_ii = ai'*ai/T;
s_jj = aj'*aj/T;
s_ij = ai'*aj/T;
p_ij = 1/T*sum((ai.*aj-s_ij).^2);
t_iiij = 1/T*sum((ai.^2-s_ii).*(ai.*aj-s_ij));
t_jjij = 1/T*sum((aj.^2-s_jj).*(ai.*aj-s_ij));