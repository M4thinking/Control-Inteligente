% Función objetivo modelo neuronal
function J = objective_nn(yprev, uprev, uprox, model, r, n)
    L = 0.00001;
    % yprev = [y(k-1), y(k-2), ..., y(k-nreg)]
    % uprev = [u(k-1), u(k-2), ..., u(k-nreg)]
    % uprox = [u(k+n-1), u(k+8), ..., u(k)]
    % du(k+j-1) = u(k+j-1)-u(k+j-2). Ej con j = 1: du(k) = u(k) - u(k-1)
    % du = flip(uprox - [uprox(2:end); uprev(1)])
    %    = flip[u(k+9) - u(k+8), u(k+8) - u(k+7), ..., u(k) - u(k-1)]
    %    = flip[du(k+9), du(k+8), ..., du(k)]
    %    = [du(k), ..., du(k+8), du(k+9)]
    % du empiezan de 0, pero matlab no indexa así.
    du = flip(uprox - [uprox(2:end); uprev(1)]);
    % Calculamos J con regresores anteriores a cada y(k+j)
    regy = yprev;
    regu = uprev;
    J = 0;
    for t=1:n
        y = my_ann_evaluation(model, [regy; regu]);
        J = J + (y - r(t))^2 + L*(du(t))^2;
        % Usar las predicciones para los regresores tq du(t>k) e y(t > k-1) 
        regy = [y; regy(1:end-1)];
        regu = [uprox(t); regu(1:end-1)];
    end
end