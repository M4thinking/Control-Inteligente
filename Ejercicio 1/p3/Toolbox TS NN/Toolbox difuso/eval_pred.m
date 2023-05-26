function [y_pred, z] = eval_pred(z, model, regs, n_pred)
    % Predicciones a 8 pasos con forloop para ysim. x_optim_val = [yk-1, yk-2, ..., yk-Ny, uk-1, uk-2, ..., uk-Nu] 
    uk = z(:, regs+1); % uk = [uk-1, uk-2]
    y_pred = zeros(length(z),1); % [yk, yk+1, ..., yk+n_pred-1]
    % Evaluar 8 veces el modelo para obtener predicciones hasta yk+7|k-1 y luego hasta yk+n_pred-1|k-1
    for i=1:n_pred
        y_hat = ysim(z, model.a, model.b, model.g);
        y_pred = [zeros(regs, 1); y_hat]; % % Rellenamos con 0 en la última posición (no sabemos u(k+i) para k>Nd)
        if i == n_pred
            break
        end
        % Ponemos que en uk, uk+1, ..., uk+7 sean ceros (no serán usados)
        [~, z] = autoregresores([zeros(i, 1); uk(1:end-i)],y_pred, regs); % x_optim_val_pred = [yk-1, yk-2, uk-1, uk-2]
    end
    y_pred = y_pred(regs+1:end-n_pred);
    z = z(n_pred+1:end,:);
end