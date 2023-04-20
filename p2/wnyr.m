function [Wn_tot, yr_tot]=wnyr(X,a,b,g)
% Creates the model's predicction
% y is the vector of outputs when evaluating the TS defined by a,b,g
% X is the data matrix
% a is the cluster's Std^-1 
% b is the cluster's center
% g is the consecuence parameters

% Nd number of point we want to evaluate
% n is the number of regressors of the TS model

[Nd,n]=size(X);

% NR is the number of rules of the TS model
NR=size(a,1);         
y=zeros(Nd,1);
Wn_tot = [];
yr_tot = [];
     
for k=1:Nd 
    
    % W(r) is the activation degree of the rule r
    % mu(r,i) is the activation degree of rule r, regressor i
    W=ones(1,NR);
    mu=zeros(NR,n);
    for r=1:NR
     for i=1:n
       mu(r,i)=exp(-0.5*(a(r,i)*(X(k,i)-b(r,i)))^2);  
       W(r)=W(r)*mu(r,i);
     end
    end

    % Wn(r) is the normalized activation degree
    if sum(W)==0
        Wn=W;
    else
        Wn=W/sum(W);
    end
    
    % Now we evaluate the consequences
   
    yr=g*[1 ;X(k,:)'];  
    
    
    Wn_tot = [Wn_tot;Wn];
    
    yr_tot = [yr_tot;yr'];
    
    % Finally the output
%     if y(k) < 5
%         y(k) = 0;
%     end

end

end
