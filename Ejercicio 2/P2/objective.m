% Función objetivo
function cost = objective(u, u_prev, model, x0, ref, n,Ta,w,Ts)
    % Calcular los valores de control u
    du = diff([u_prev; u]);  % u_prev -> u en el instante anterior

    % Simulación de la planta con el modelo estimado
    X = zeros(numel(x0), n+1);  % Matriz de estado

    % Aplicar el modelo de forma iterativa
    X(:, 1) = x0;
    for i = 1:n
        X(:,i+1) = model(X(:,i),u(i),Ta,w,Ts);
    end

    % Extraer los valores predichos de la salida (theta) y la entrada (u)
    y_pred = X(1, :)';

    % Calcular la función objetivo (puede ser cualquier criterio deseado)
    cost = sum((y_pred(2:end)-ref).^2) ; 
end