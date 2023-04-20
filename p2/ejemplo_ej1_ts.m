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
[Y_val , Y_test, Y_ent, X_val, X_test, X_ent] = separar_datos(y, x, porcentajes);
%% Optimizar modelo - Reglas
[err_test, err_ent] = clusters_optimo(Y_test, Y_ent, X_test, X_ent, max_clusters);
% Criterio de Akaike
% n = length(Y_test); % Tamaño de la muestra
% AIC = [];
% for k=1:max_clusters-1
%     err = err_test(k); % Error de prueba (tomando el último valor en err_test)
%     AIC = [AIC, n*(1 + log(err)) + 2*k + (2*k*(k+1))./(n-k-1)]; % Fórmula del criterio de Akaike
%     [min_AIC, idx] = min(AIC); % Busca el valor mínimo de AIC y su índice correspondiente
%     best_clusters = idx; % El mejor número de clusters es el índice con el valor mínimo de AIC
% end
best_clusters = 2;
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')

%% Optimizar modelo - Regresores
[p, indices] = sensibilidad(Y_ent, X_ent, best_clusters); % 2 es el numero de clusters elegido anteriormente
% Elijo los regresores más importantes a partir del análisis anterior
n_regresores = 4; % Cambiar valor para mayor o menor número de regresores
[~, sorted_indices] = sort(indices, 'descend');
x_optim_ent = X_ent(:, sort(sorted_indices(1:n_regresores), 'ascend'));
x_optim_val = X_val(:, sort(sorted_indices(1:n_regresores), 'ascend'));


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
plot(t(1:len), y_hat_ent(1:len), 'r-', 'LineWidth', 1);
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