
function [y_hat, y_sup, y_inf, PICP, PINAW, J] = eval_fuzzy_nums(z, a, b, g, s, y, nu1, nu2, n3, alpha)
    % Intervalos de inceridumbre - Método de números difusos
    % El problema consiste en optimizar J = nu1 * PINAW + exp(-nu2*(PICP-(1-alpha)))
    % respecto a: s_inf(k+j) y s_sup(k+j)
    % sujeto a: PICP = (1-alpha)
    % Donde PINAW = 1/(N*R) * sum(Y_sup - Y_inf)
    % f_supj = yj + sum_i=1^Nr(su_ij * z_k) + su_0j
    % f_infj = yj + sum_i=1^Nr(sl_ij * z_k) + sl_0j
    % f_sup = ysim + sum(hj*f_supj)
    % f_inf = ysim + sum(hj*f_infj)
    % Y_sup = ysim + f_sup
    % Y_inf = ysim - f_inf
    [y_hat, h] = ysim2(z, a, b, g); % h:(Nd, NR)
    Nd = size(h,1);
    Nrules = size(h,2);
    Nregs = size(z,2);
    % s:(Nregs*2*(Nrules+1)) -> (Nregs, 2*(Nrules+1))
    s = reshape(s, Nregs, 2*(Nrules+1));
    % Inicialización de variables
    su  = s(:,1:Nrules);
    su0 = s(:,Nrules+1);
    sl  = s(:,Nrules+2:end-1);
    sl0 = s(:,end);
    f_sup = zeros(Nd,1);
    f_inf = zeros(Nd,1);

    % Aplicación de restricciones y función de costo
    for j = 1:Nrules
        % f_j:(Nd, 1) = z:(Nd, Nregs) x s:(Nregs, 1) + I*s0:(1,1)
        f_supj = abs(z)*su(:,j) + su0(j); % Solo ancho j
        f_infj = abs(z)*sl(:,j) + sl0(j); % Solo ancho j
        % f_ : (Nd,1) + (Nd,1).*(Nd,1)
        f_sup = f_sup + h(:,j).*f_supj; % Solo ancho
        f_inf = f_inf + h(:,j).*f_infj; % Solo ancho
    end
    y_sup = y_hat + f_sup;
    y_inf = y_hat - f_inf;
    R = max(y_sup) - min(y_inf);
    PICP = sum(y_inf<=y & y<=y_sup)/double(Nd);
    PINAW = 1/double(Nd*R) * sum(y_sup-y_inf)*100;
    % Función de costo (Regularización)
    J = nu1*PINAW + exp(-nu2*(PICP - (1-alpha))*100) + n3*sum(s(:).^2);
end