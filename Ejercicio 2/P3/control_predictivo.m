function [u_next, u0] = control_predictivo(n, model, u0, x0, u_prev, ref)    
    % Restricciones de igualdad y desigualdad
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb = -40 * ones(n, 1);  % Límite inferior de u
    ub = 40 * ones(n, 1);   % Límite superior de u
    dumin = -30;
    dumax = 30;
    % Llamada a fmincon para optimizar la función objetivo
    nonlcon = @(u) nonlcon_cart(u_prev, u, dumin, dumax, n);
    options = optimoptions('fmincon', 'Display', 'off');
    u_opt = fmincon(@(u) objective(u, u_prev, model, x0, ref, n), u0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    
    % Valor de control a aplicar en el siguiente instante
    u_next = u_opt(1);
    u0 = [u_opt(2:end);u_opt(end)]; % Apuesta razonable para el siguiente instante
end

