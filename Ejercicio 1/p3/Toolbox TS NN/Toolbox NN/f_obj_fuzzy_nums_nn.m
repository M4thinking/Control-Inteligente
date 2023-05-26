function [J] = f_obj_fuzzy_nums_nn(z, net, s, y, nu1, nu2,nu3, alpha)
    [~,y_sup,y_inf,PICP,PINAW,J] = eval_fuzzy_nums_nn(z, net, s, y, nu1, nu2,nu3, alpha);
    if abs(PICP-(1-alpha)) <= 1e-3
        disp('Detalle');
        disp(PICP);
        disp(max(y_sup-y_inf));
        disp(PINAW);
    end
end