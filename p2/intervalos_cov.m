function [y_pred, I] = intervalos_cov(z, model, stds, K)
    % Se obtiene estimación de modelo T-S y su intervalo
    [y_pred, h] = ysim2(z, model.a, model.b, model.g); %h:(Nd, NR)
    Nrules = size(stds,1);
    Nd = size(z,1);
    dy = zeros(Nd, Nrules);
    for j = 1:Nrules
        for i = 1:Nd
            Pij = h(i,j)*z(i,:)';
            dy(i,j) = stds(j)*sqrt(1 + Pij'*K(:,:,j)*Pij);
        end
    end
    I = zeros(Nd,1);
    for j = 1:Nrules
        I = I + h(:,j).*dy(:,j);
    end
end