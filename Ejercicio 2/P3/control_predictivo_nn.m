function u = control_predictivo_nn(yprev,uprev,xprev,modelx, modely,r,n)
% uprox0 es la condición inicial puede ser cualquier valor
uprox0 = repelem(0,n)';
% Se miniza respecto a u. fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon)
% Restricciones de igualdad y desigualdad
A = [];
b = [];
Aeq = [];
beq = [];
lb = -250 * ones(n, 1);  % Límite inferior de u
ub = 250 * ones(n, 1);   % Límite superior de u
dumin = -100;
dumax = 100;
% Llamada a fmincon para optimizar la función objetivo
nonlcon = @(u) nonlcon_cart_nn(xprev, uprev, yprev, u, dumin, dumax, modelx, modely, n);
options = optimoptions('fmincon', 'Display', 'off');
uprox = fmincon(@(uprox)objective_nn(yprev, uprev, uprox, modely, r, n), uprox0, A, b, Aeq, beq, lb, ub, [], options);
% uprox = [u(k+n-1), u(k+8), ..., u(k)]
u = uprox(end);
end