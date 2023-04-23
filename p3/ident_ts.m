% clear
clc
%% Definicion de condiciones iniciales y aprbs 

y0 = [0; 0; pi-0.1; 0];
aprbs=aprbsGen(); %60 segundos con una amplitud de [pi+0.1,pi-0.1], ts=0.1, tau=6
%% Simulacion

sim('ident_pendcart.slx')

%% Identificacion

max_regs = 4;
max_regs_list = 1:max_regs;
max_clusters = 15;

% estado
x=salida(:,1);
dx=salida(:,2);
theta=salida(:,3);
dtheta=salida(:,4);

porcentajes=[0.6,0.2,0.2];
[y ,x]=autoregresores(entrada,theta,max_regs);
[Y_test , Y_val, Y_ent, X_test, X_val, X_ent] = separar_datos(y, x, porcentajes);

%% Optimizar modelo - Reglas
[err_test, err_ent] = clusters_optimo(Y_val, Y_ent, X_val, X_ent, max_clusters);
best_clusters = 4; % Criterio del codo
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')
title('Error en Función del Número de Reglas');
xlabel('Número de Reglas')
ylabel('Error Cuadrático Medio')

%% Optimizar modelo - Regresores
[p, indices] = sensibilidad(Y_ent, X_ent, best_clusters); % 4 es el numero de clusters elegido anteriormente
n_regresores = 3; % Cambiar valor para mayor o menor número de regresores
best_indices = [];
for i=1:n_regresores % Descartamos peor regresor
    [~, idx] = max(indices);
    best_indices = [best_indices, idx];
    indices(idx) = 0;
end

x_optim_ent = X_ent(:, sort(best_indices, 'ascend'));
x_optim_val = X_val(:, sort(best_indices, 'ascend'));

%% Entrenar modelo
[model, ~] = TakagiSugeno(Y_ent, x_optim_ent, best_clusters, [1 2 2]);

%% Predicciones
y_hat = ysim(x_optim_val, model.a, model.b, model.g);

figure()
plot(Y_val, '.b')
hold on
plot(y_hat, 'r')

legend('Valor real', 'Valor esperado')
xlabel('Tiempo')
ylabel('Salida')
hold off
%% 

%% Evaluar predicciones a 8 y 16 pasos
regs = 2;
% Predicciones a 8 pasos con forloop para ysim. x_optim_val = [yk-1, yk-2, ..., yk-Ny, uk-1, uk-2, ..., uk-Nu] 
uk = x_optim_val(:, regs+1); % uk = [uk-1, uk-2]
yk = zeros(length(x_optim_val), 16); % [yk, yk+1, ..., yk+7]
% Evaluar 8 veces el modelo para obtener predicciones hasta yk+7|k-1 y luego hasta yk+15|k-1

[y, x_optim_val_pred] = autoregresores(uk, Y_val, regs); % x_optim_val_pred = [yk-1, yk-2, uk-1, uk-2]
for i=1:16
    disp(i);
    % yk = [zeros; ysim]
    yy = [zeros(regs, 1); ysim(x_optim_val_pred, model.a, model.b, model.g)]; % yk = [yk, yk+1, ..., yk+7]
    yk(:, i) = yy;
    % Ponemos que en uk, uk+1, ..., uk+7 sean ceros
    [y, x_optim_val_pred] = autoregresores([zeros(i, 1); uk(1:end-i)],yk(:, i), regs); % x_optim_val_pred = [yk-1, yk-2, uk-1, uk-2]
end

y_hat_8 = yk(1:end-8, 8); % Predicciones a 8 pasos
y_hat_16 = yk(1:end-16, 16); % Predicciones a 16 pasos

%% 
% Graficar predicciones a 8 y sus intervalos de incertidumbre
figure()
plot(Y_val, '.b')
hold on;
% fill y_hat_8_sup y y_hat_8_inf
% t = 1:length(Y_val);
% t2 = [t, fliplr(t)];
% inBetween = [y_hat_8_sup; flipud(y_hat_8_inf)];
% fill(t2, inBetween, [0.5 0.5 1], 'FaceAlpha', 0.5);
% hold on;
plot(y_hat_8, 'r')
hold on;
plot(y_hat_16, 'g')
legend('Valor real', 'y_hat_8', 'y_hat_16')
 
