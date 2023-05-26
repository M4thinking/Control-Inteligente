function [yk, I_pred] = eval_pred_cov(z_pred, y, model, std_ent, K, regs, n_pred)
    % Predicciones a 8 pasos con forloop para ysim. x_optim_val = [yk-1, yk-2, ..., yk-Ny, uk-1, uk-2, ..., uk-Nu] 
    uk = z_pred(:, regs+1); % uk = [uk-1, uk-2]
    yk = zeros(length(z_pred), n_pred); % [yk, yk+1, ..., yk+n_pred-1]
    % Evaluar 8 veces el modelo para obtener predicciones hasta yk+7|k-1 y luego hasta yk+n_pred-1|k-1
    I_pred = zeros(length(z_pred),n_pred);
    [~, z_pred] = autoregresores(uk, y, regs); % z_pred = [y_hat(k-1+i), y_hat(k-2+i), u(k-1+i), u(k-2+i)]
    for i=1:n_pred
        [y_hati, Ii] = intervalos_cov(z_pred, model.a, model.b, model.g, std_ent, K);
        Ii = [zeros(regs, 1); Ii]; % Rellenamos con 0 en la última posición (no sabemos u(k+i) para k>Nd)
        I_pred(:,i) = Ii;
        y_hati = [zeros(regs, 1); y_hati]; % % Rellenamos con 0 en la última posición (no sabemos u(k+i) para k>Nd)
        yk(:, i) = y_hati;
        % Ponemos que en uk, uk+1, ..., uk+7 sean ceros (no serán usados)
        [~, z_pred] = autoregresores([zeros(i, 1); uk(1:end-i)],yk(:, i), regs); % x_optim_val_pred = [yk-1, yk-2, uk-1, uk-2]
    end
end