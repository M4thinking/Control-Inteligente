function [K, stds] = get_cov_params(z,y, model)
    [y_hat, h] = ysim2(z, model.a, model.b, model.g);
    % Obtenemos los Phi (P), Pi,j = h(i,j)*x(i,:)^T
    [lenz, Nregs] = size(z);
    Nrules = size(h,2);
    K = zeros(Nregs, Nregs, Nrules);
    for j=1:Nrules
        Pj = zeros(Nregs, length(z));
        for i=1:lenz
            Pij = h(i,j)*z(i,:)';
            Pj(:,i) = Pij;
        end
        K(:,:,j) = (Pj*Pj')^-1;
    end
    % Calculamos el error ej = yj - y_hatj (yj = y*hj)
    e_ent = zeros(lenz, Nrules);
    stds = zeros(Nrules,1);
    for j=1:Nrules
        y_hatj = y_hat.*h(:,j);
        yj = y.*h(:,j);
        e_ent(:,j) = yj - y_hatj;
        stds(j) = std(e_ent(:,j));
    end
end