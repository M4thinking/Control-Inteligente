function [c,ceq] = nonlcon_cart_nn(x_prev, u_prev,u,dumin, dumax,model, n)
    cmin = ones(n,1)*dumin - (u(1:end)-[u_prev(1); u(1:end-1)]);
    cmax = u(1:end)-[u_prev(1); u(1:end-1)] - ones(n,1)*dumax;
    regx = x_prev;
    regu = u_prev;
    c = [];
    for t=1:n
        y = my_ann_evaluation(model, [regx; regu]');
        c = [c; y -2*pi; -y-2*pi]; %y-2*pi <= 0 , -y-2*pi <= 0
        regy = [y; regy(1:end-1)];
        regu = [uprox(t); regu(1:end-1)];
    end
    c = [c; cmin; cmax];
    ceq = 0*x_prev; % FALTAN RESTRICCIONES DEL ESTADO
end