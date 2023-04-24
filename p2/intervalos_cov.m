function [y_hat, I] = intervalos_cov(z, a, b, g, std, K)
    % Se obtiene estimación de modelo T-S y su intervalo
    [y_hat, h] = ysim2(z, a, b, g); %h:(Nd, NR)
    rules = size(std,1);
    Nd = size(z,1);
    dy = zeros(Nd, rules);
    for j = 1:rules
        for i = 1:Nd
            Pij = h(i,j)*z(i,:)';
            dy(i,j) = std(j)*sqrt(1 + Pij'*K(:,:,j)*Pij);
        end
    end
    I = zeros(Nd,1);
    for j = 1:rules
        I = I + h(:,j).*dy(:,j);
    end
end