function [c,ceq] = nonlcon_cart_nn(x_prev, u_prev, y_prev,u,dumin, dumax,modelx,modely, n)
    cmin = ones(n,1)*dumin - (u(1:end)-[u_prev(1); u(1:end-1)]);
    cmax = u(1:end)-[u_prev(1); u(1:end-1)] - ones(n,1)*dumax;
    %regx = x_prev;
    regy = y_prev;
    regu = u_prev;
    c = [];
    for t=1:n
        %x = my_ann_evaluation(modelx, [regx; regu]);
        y = my_ann_evaluation(modely, [regy; regu]);
        %c = [c; (x-100); (-x-100)];
        c = [c; (y-2*pi); (-y-2*pi)]; %y-2*pi <= 0 , -y-2*pi <= 0
        %regx = [x; regx(1:end-1)];
        regy = [y; regy(1:end-1)];
        regu = [u(t); regu(1:end-1)];
    end
    c = [c; cmin; cmax];
    ceq = 0;
end