function [J] = fobj_fuzzy_nums(z, a, b, g, s, y, nu1, nu2, n3, alpha)
    [~,y_sup,y_inf,PICP,PINAW,J] = eval_fuzzy_nums(z, a, b, g, s, y, nu1, nu2, n3, alpha);
    if abs(PICP-0.9) <= 1e-3 
        disp('Detalle');
        disp(PICP);
        disp(max(y_sup-y_inf));
        disp(PINAW);
    end
end