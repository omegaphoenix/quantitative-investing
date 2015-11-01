function debug2()
beta=rand(100,1);
nb2=50;
sigma=eye(100);
lambda=1.0;

cvx_begin
    variable  wL(nb2,1) nonnegative
    variable  wS(nb2,1) nonnegative
    w = [wL;-wS];
    minimize  (w'*sigma*w)    
cvx_end
1;