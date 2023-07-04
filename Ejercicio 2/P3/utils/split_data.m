function [Y_train, X_train, Y_val, X_val, Y_test, X_test] = split_data(Y, X, train_ratio, val_ratio, test_ratio)
    % Split data en train, validation and test
    % Cantidad de datos
    N = size(Y, 2);
    N_train = floor(N * train_ratio);
    N_val = floor(N * val_ratio);
    N_test = floor(N * test_ratio);
    % Datos de entrenamiento
    Y_train = Y(:, 1:N_train);
    X_train = X(:, 1:N_train); 
    Y_val = Y(:, N_train+1:N_train + N_val);
    X_val = X(:, N_train+1:N_train + N_val);
    Y_test = Y(:, N_train + N_val + 1:end);
    X_test = X(:, N_train + N_val + 1:end);
end