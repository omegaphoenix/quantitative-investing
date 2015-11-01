function debug1()
%%
load test.mat

N=size(test,2);

S = zeros(N,N);
PI = S; Tiiij = S; Tjjij=S;

for i=1:N
    for j=1:N        
        [PI(i,j), S(i,j),Tiiij(i,j),Tjjij(i,j)] = calc(test(:,i),test(:,j));
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

rdiag = sum(diag(PI));
roff = 0;
for i=1:N
    for j=1:N
        if(j~=i)
            roff = roff+1/2*(sqrt(S(j,j)/S(i,i))*Tiiij(i,j)+sqrt(S(i,i)/S(j,j))*Tjjij(i,j)); 
        end
    end
end

spi = sum(PI(:));
gamma = sum(sum( (F-S).^2));

rho = rdiag + Rbar*roff;
k = (spi-rho)/gamma;
k/size(test,1)



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


