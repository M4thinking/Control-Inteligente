function [u_next, u0] = control_predictivo(n, model, u0, x0, u_prev, ref,Ta,w,Ts)    
    % Restricciones de igualdad y desigualdad
    A = [];
    b = [];
    Aeq = [];
    beq = [];
    lb =0.1 * ones(n, 1);  % Límite inferior de u
    ub = 2 * ones(n, 1);   % Límite superior de u
    dumin = -20;
    dumax = 20;
    % Llamada a fmincon para optimizar la función objetivo
    nonlcon = @(u) nonlcon_cart(u_prev, u, dumin, dumax, n);
    options = optimoptions('fmincon', 'Display', 'off');
    u_opt = fmincon(@(u) objective(u, u_prev, model, x0, ref, n,Ta,w,Ts), u0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    
    % Valor de control a aplicar en el siguiente instante
    u_next = u_opt(1);
    u0 = [u_opt(2:end);u_opt(end)]; % Apuesta razonable para el siguiente instante
end

