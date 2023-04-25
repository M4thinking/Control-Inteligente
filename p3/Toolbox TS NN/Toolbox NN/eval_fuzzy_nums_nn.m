function [y_hat, y_sup, y_inf, PICP, PINAW, J] = eval_fuzzy_nums_nn(z, net, s, y, nu1, nu2, nu3, alpha)
    % Intervalos de inceridumbre - Método de números difusos
    % El problema consiste en optimizar J = nu1 * PINAW + exp(-nu2*(PICP-(1-alpha)))
    % respecto a: s_inf(k+j) y s_sup(k+j)
    % sujeto a: PICP = (1-alpha)
    % Donde PINAW = 1/(N*R) * sum(Y_sup - Y_inf)
    input = z'; % h:(Nd, NR)
    % Works with only single INPUT vector
    % Matrix version can be implemented
    ymax = net.input_ymax;
    ymin = net.input_ymin;
    xmax = net.input_xmax;
    xmin = net.input_xmin;
    input_preprocessed = (ymax-ymin) * (input-xmin) ./ (xmax-xmin) + ymin;
    % Pass it through the ANN matrix multiplication
    y1 = tanh(net.IW * input_preprocessed + net.b1);
    y2 = net.LW * y1 + net.b2;
    ymax = net.output_ymax;
    ymin = net.output_ymin;
    xmax = net.output_xmax;
    xmin = net.output_xmin;
    y_hat = ((y2-ymin) .* (xmax-xmin) /(ymax-ymin) + xmin)';
    Nregs = size(z,2);
    % s:(Nregs*2*(Nrules+1)) -> (Nregs, 2*(Nrules+1))
    s = reshape(s, Nregs+1, []);
    Nd = size(y,1);
    Nrules = size(y1,1);
    % s:(Nregs*2*(Nrules+1)) -> (Nregs, 2*(Nrules+1))
    % Inicialización de variables
    su  = s(1:end-1,1:Nrules);
    su0 = s(end,1:Nrules);
    sl  = s(1:end-1,Nrules+1:end);
    sl0 = s(end,Nrules+1:end);
    f_sup = zeros(Nd,1);
    f_inf = zeros(Nd,1);
    % Aplicación de restricciones y función de costo
    for j = 1:Nrules
        % f_j:(Nd, 1) = z:(Nd, Nregs) x s:(Nregs, 1) + I*s0:(1,1)
        f_supj = abs(z)*su(:,j) + su0(j); % Solo ancho j
        f_infj = abs(z)*sl(:,j) + sl0(j); % Solo ancho j
        % f_ : (Nd,1) + (Nd,1).*(Nd,1)
        f_sup = f_sup + net.LW(:,j).*f_supj; % Solo ancho
        f_inf = f_inf + net.LW(:,j).*f_infj; % Solo ancho
    end
    f_sup = ((f_sup-ymin) .* (xmax-xmin) /(ymax-ymin) + xmin);
    f_inf = ((f_inf-ymin) .* (xmax-xmin) /(ymax-ymin) + xmin);
    y_sup = y_hat + abs(f_sup);
    y_inf = y_hat - abs(f_inf);
    PICP = calc_picp(y, y_inf, y_sup);
    PINAW = calc_pinaw(y, y_inf, y_sup);
    % Función de costo (Regularización)
    % su_ij^2 + sl_ij^2 + su0_j^2 + sl0_j^2
    sum_s2 = sum(sum(su.^2)) + sum(sum(sl.^2)) + sum(su0.^2) + sum(sl0.^2);
    J = nu1*PINAW + exp(-nu2*(PICP - (1-alpha))) + nu3*sum_s2;
end