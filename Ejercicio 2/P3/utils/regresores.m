function regs = regresores(x, n)
N=length(x);regs=zeros(N-n,n);for i=1:n; regs(:,i)=x(n-i+1:N-i);end
end
