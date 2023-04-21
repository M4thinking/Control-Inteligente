clear
clc

addpath("Toolbox TS NN/Toolbox difuso")
%% Generación APRBS
aprbs = aprbsGen();
%% Correr simulink
out = sim('ident_model.slx');
%% Cargar datos
% load('datos_ejemplo.mat')
% 
% % Se eliminan datos NaN, necesario solo para estos datos
% nanx = isnan(datos);
% t    = 1:numel(datos);
% datos(nanx) = interp1(t(~nanx), datos(~nanx), t(nanx));
% 
% figure()
% plot(datos)
% title('Temperatura ambiental')
% xlabel('Muestras')
% ylabel('Temperatura [°C]')
%% Parametros modelo
max_regs = 5;
max_regs_list = 1:max_regs;
max_clusters = 16;

% Se cargan el vector Y de salida y la matriz X de regresores
% Recordar que el orden de Y,X fue elegido arbitrariamente y su forma
% dependera de cada implementacion

% [y, x] = autoregresores(datos, max_regs);
% 
% [Y.val , Y.test, Y.ent, X.val, X.test, X.ent] = separar_datos(y, x, porcentajes);

porcentajes=[0.6,0.2,0.2];
[y ,x]=autoregresores(out.entrada,out.salida,max_regs);
[Y_test , Y_val, Y_ent, X_test, X_val, X_ent] = separar_datos(y, x, porcentajes);
%% Optimizar modelo - Reglas
[err_test, err_ent] = clusters_optimo(Y_val, Y_ent, X_val, X_ent, max_clusters);
best_clusters = 2; % Criterio del codo
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')

%% Optimizar modelo - Regresores
[p, indices] = sensibilidad(Y_ent, X_ent, best_clusters); % 2 es el numero de clusters elegido anteriormente
n_regresores = 4; % Cambiar valor para mayor o menor número de regresores
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

%% Modelo con intervalo de incertidumbre - Método de la covarianza
% Se obtiene la matriz de covarianza cov(yr_ent - yr_hat)
[y_hat_val, wj_hat_val, yj_hat_val] = wnyr(x_optim_val, model.a, model.b, model.g);
yj_val = wj_hat_val.*Y_val;
delta_yj = sqrt(cov(yj_val - yj_hat_val));

% Intervalo de incertidumbre para cada regla en entrenamiento
alphas = [];
y_sups = []; % Limite superior intervalo
y_infs = []; % Limite inferior intervalo
for porcentaje=1:9
    alpha = 0;
    n_total = 0;
    while n_total < porcentaje/10.0
        [y_hat, y_sup, y_inf] = intervalos_cov(x_optim_ent, model.a, model.b, model.g, delta_yj, alpha);
        alpha = alpha + 0.001;
        n_total = sum(y_inf<=Y_ent & Y_ent<=y_sup)/double(sum(length(Y_ent)));
    end
    y_sups = [y_sups, y_sup];
    y_infs = [y_infs, y_inf];
    alphas = [alphas, alpha];
end
%---------------------------------

t = 1:length(Y_ent);
len = 1000; % Cambiar a length(Y_ent) para ver todos los datos.
figure();
% Graficar intervalos de incertidumbre
for i = flip(1:9)
    alpha = alphas(i);
    y_sup = y_sups(1:len,i);
    y_inf = y_infs(1:len,i);
    t2 = [t(1:len), fliplr(t(1:len))];
    inBetween = [y_sup; flipud(y_inf)];
    fill(t2, inBetween, [0.5 (1-i/10.0) 1], 'FaceAlpha', (10-i)/12.0);
    hold on;
end

% Graficar curva de estimación y_hat_ent
plot(t(1:len), y_hat(1:len), 'r-', 'LineWidth', 1);
hold on;
% Graficar puntos Y_ent
scatter(t(1:len), Y_ent(1:len), 5, 'b', 'filled');
hold on;
% Configuración de la gráfica
xlabel('Variable independiente');
ylabel('Variable dependiente');
title('Modelo con intervalo de incertidumbre - Método de la covarianza');
legend('90%','80%','70%', '60%','50%', '40%','30%','20%','10%',...
    'Estimación', 'Datos de entrenamiento', 'Intervalos de incertidumbre');

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
 

 

 