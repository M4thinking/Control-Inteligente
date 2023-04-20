function [y_hat, y_sup, y_inf] = intervalos_cov(x, a, b, g, delta_yj, alpha)
    % Se obitenen modelos T-S de entranmiento para sintonizar alpha
    [y_hat, wj_hat, yj_hat] = wnyr(x, a, b, g);
    % Intervalo de incertidumbre para cada regla en entrenamiento
    f_supj =  yj_hat + alpha*diag(delta_yj)';
    f_infj =  yj_hat - alpha*diag(delta_yj)';
    % Intervalo de incertidumbre total
    f_sup = sum(wj_hat.*f_supj,2);
    f_inf = sum(wj_hat.*f_infj,2);
    I = sum(wj_hat.*diag(delta_yj)',2);
    y_sup = f_sup + alpha*I;
    y_inf = f_inf - alpha*I;
end