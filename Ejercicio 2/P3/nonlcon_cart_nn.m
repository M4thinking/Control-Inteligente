function [c,ceq] = nonlcon_cart_nn(x_prev, u_prev,u,dumin, dumax, n)
    cmin = ones(n,1)*dumin - (u(1:end)-[u_prev; u(1:end-1)]);
    cmax = u(1:end)-[u_prev; u(1:end-1)] - ones(n,1)*dumax;
    c = [cmin; cmax];
    ceq = 0*x_prev; % FALTAN RESTRICCIONES DEL ESTADO
end