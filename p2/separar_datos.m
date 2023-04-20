function [Y_val, Y_test, Y_train, X_val, X_test, X_train] = separar_datos(Y,X, porcentajes)
% X: matriz de autoregresores
% Y: vector de salida
% porcentajes: lista de porcentajes [train, val, test] que indican la proporción de datos para cada conjunto

% Calcular el número de filas en la matriz X y el vector Y
N = size(X, 1);

% Calcular el número de filas para cada conjunto
num_train = round(porcentajes(1) * N);
num_val = round(porcentajes(2) * N);
num_test = round(porcentajes(3) * N);

% Particionar los datos en conjuntos de entrenamiento, validación y prueba
X_train = X(1:num_train,:);
Y_train = Y(1:num_train);
X_val = X(num_train+1:num_train+num_val,:);
Y_val = Y(num_train+1:num_train+num_val);
X_test = X(num_train+num_val+1:end,:);
Y_test = Y(num_train+num_val+1:end);

end
