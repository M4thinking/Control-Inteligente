clear
clc

addpath("../Toolbox difuso")
%% Cargar datos
load('datos_ejemplo.mat')

% Se eliminan datos NaN, necesario solo para estos datos
nanx = isnan(datos);
t    = 1:numel(datos);
datos(nanx) = interp1(t(~nanx), datos(~nanx), t(nanx));

figure()
plot(datos)
title('Temperatura ambiental')
xlabel('Muestras')
ylabel('Temperatura [°C]')
%% Parametros modelo
max_regs = 144;
max_regs_list = 1:max_regs;
max_clusters = 10;
porcentajes = [20 20 60];

% Se cargan el vector Y de salida y la matriz X de regresores
% Recordar que el orden de Y,X fue elegido arbitrariamente y su forma
% dependera de cada implementacion

% [y, x] = autoregresores(datos, max_regs);
% 
% [Y.val , Y.test, Y.ent, X.val, X.test, X.ent] = separar_datos(y, x, porcentajes);
load("autorregresores.mat")


%% Optimizar modelo - Reglas
[err_test, err_ent] = clusters_optimo(Y.test, Y.ent, X.test, X.ent, max_clusters);
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')

%% Optimizar modelo - Regresores
[p, indices] = sensibilidad(Y.ent, X.ent, 2); % 2 es el numero de clusters elegido anteriormente

% Elijo los primeros 3 regresores a partir del analisis anterior
x_optim_ent = X.ent(:, [1 2 3]);
x_optim_val = X.val(:, [1 2 3]);


%% Entrenar modelo
[model, ~] = TakagiSugeno(Y.ent, x_optim_ent, 2, [1 2 2]);

%% Predicciones
y_hat = ysim(x_optim_val, model.a, model.b, model.g);

figure()
plot(Y.val, '.b')
hold on
plot(y_hat, 'r')

legend('Valor real', 'Valor esperado')

