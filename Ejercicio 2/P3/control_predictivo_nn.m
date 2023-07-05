function u = control_predictivo_nn(yprev,uprev,xprev,model,r,n)
% uprox0 es la condición inicial puede ser cualquier valor
uprox0 = repelem(uprev(1),n)';
% Se miniza respecto a u. fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon)
% Restricciones de igualdad y desigualdad
A = [];
b = [];
Aeq = [];
beq = [];
lb = -50 * ones(n, 1);  % Límite inferior de u
ub = 50 * ones(n, 1);   % Límite superior de u
dumin = -20;
dumax = 20;
% Llamada a fmincon para optimizar la función objetivo
%nonlcon = @(u) nonlcon_cart_nn(xprev, uprev', u, dumin, dumax, n);
options = optimoptions('fmincon', 'Display', 'off');
uprox = fmincon(@(uprox)objective_nn(yprev, uprev, uprox, model, r, n), uprox0, A, b, Aeq, beq, lb, ub, [], options);
% uprox = [u(k+9), u(k+8), ..., u(k)]
u = uprox(end);
end