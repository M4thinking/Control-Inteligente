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
% Se obtiene la matriz de covarianza de los datos de validación
[wn_hat, yr_hat] = wnyr(x_optim_val, model.a, model.b, model.g);
%cov_mat = cov(-y_hat);

% Se obtienen los pesos de la red neuronal
%weights = net_optim.IW{1};

% Se calcula la varianza del modelo
%variance = diag(x_optim_val * cov_mat * x_optim_val');

% % Se calcula el intervalo de incertidumbre para cada punto de validación
% interval = 1.96 * sqrt(variance);
% 
% % Se grafica el modelo junto con el intervalo de incertidumbre
% figure()
% plot(Y_val, '.b')
% hold on
% plot(y_hat, 'r')
% plot(y_hat + interval, '--g')
% plot(y_hat - interval, '--g')
% legend('Valor real', 'Valor esperado', 'Intervalo de incertidumbre')

