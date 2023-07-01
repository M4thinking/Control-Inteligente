% Función objetivo
function cost = objective(u_prox, a,b,g, x0, r, Ta_prev,Ts)
    n = 5;
    % Obtener Ta_vec, vector de predicciones de Ta
    Ta_vec = zeros(n,1);
    for i = 1:n
        Ta_vec(i) = ysim(Ta_prev, a, b, g);
        % Borrar Ta_prev mediante un shift
        Ta_prev = [Ta_vec(i) Ta_prev(1:end-1)];
    end
    % Simulación de la planta con el modelo estimado
    X = zeros(numel(x0), n+1);  % Matriz de estado

    % Aplicar el modelo de forma iterativa
    X(:, 1) = x0;
    for i = 1:n
        X(:,i+1) = HVAC_dis(X(:,i),u_prox(i),Ta_vec(i), [0 0],Ts);
    end

    % Extraer los valores predichos de la salida (theta) y la entrada (u)
    y_pred = X(1, :)';
    ref = r*ones(n,1);
    % Calcular la función objetivo (puede ser cualquier criterio deseado)
    cost = sum((y_pred(2:end)-ref).^2) ; 
end