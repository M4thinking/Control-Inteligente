function [c,ceq] = nonlcon_cart(u_prev,u,dumin, dumax, n)
    cmin = ones(n,1)*dumin - (u(1:end)-[u_prev; u(1:end-1)]);
    cmax = u(1:end)-[u_prev; u(1:end-1)] - ones(n,1)*dumax;
    c = [cmin; cmax];
    ceq = 0;
end