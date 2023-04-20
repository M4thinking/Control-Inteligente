function [Y, X] = autoregresores(entrada,salida,n)
%max_rgs: número de autoregresores a generar

N=length(entrada);

% Crear la matriz de regresión X
X=zeros(N-n,n*2);

for i=1:n
    X(:,i)=salida(n-i+1:N-i)';
    X(:,i+n)=entrada(n-i+1:N-i);
end

Y = salida(n+1:N);

end

