clear; clc; addpath("Toolbox TS NN/Toolbox difuso")
%% Generación APRBS
y0 = [0; 0; 0; 0];
Tfinal=10000;
Ts=1;
aprbs = aprbsGen(Tfinal,Ts);
%% Correr simulink
sim('ident_pendcart.slx');
%% Parametros modelo
max_regs =10;
max_regs_list = 1:max_regs;
max_clusters = 20;

% Se cargan el vector Y de salida y la matriz X de regresores
% Recordar que el orden de Y,X fue elegido arbitrariamente y su forma
% dependera de cada implementacion

% [y, x] = autoregresores(datos, max_regs);
% 
% [Y.val , Y.test, Y.ent, X.val, X.test, X.ent] = separar_datos(y, x, porcentajes);
x=salida(:,1);
dx=salida(:,2);
theta=salida(:,3);
dtheta=salida(:,4);
plot(sin(theta));

porcentajes=[0.6,0.2,0.2];
[y ,x] = autoregresores(entrada,sin(theta),max_regs);
[Y_val , Y_test, Y_ent, X_val, X_test, X_ent] = separar_datos(y, x, porcentajes);
%% Optimizar modelo - Reglas
[err_test, err_ent] = clusters_optimo(Y_test, Y_ent, X_test, X_ent, max_clusters);
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')
title('Error en Función del Número de Reglas');
xlabel('Número de Reglas')
ylabel('Error Cuadrático Medio')
%% Optimizar modelo - Regresores
rules = 12; % Criterio anterior
[p, indices] = sensibilidad(Y_ent, X_ent,rules); % rules = numero de clusters
n_regresores = 10; % Cambiar valor para mayor o menor número de regresores
best_indices = [];
for i=1:n_regresores % Descartamos peor regresor
    [~, idx] = max(indices);
    best_indices = [best_indices, idx];
    indices(idx) = 0;
end

x_optim_ent = X_ent(:, sort(best_indices, 'ascend'));
x_optim_test = X_test(:, sort(best_indices, 'ascend'));
x_optim_val = X_val(:, sort(best_indices, 'ascend'));

%% Entrenar modelo
[model, ~] = TakagiSugeno(Y_ent, x_optim_ent, rules, [1 2 2]);

%% Predicciones
y_hat_ent = ysim(x_optim_ent, model.a, model.b, model.g);
y_hat_test = ysim(x_optim_test, model.a, model.b, model.g);
y_hat_val = ysim(x_optim_val, model.a, model.b, model.g);

figure()
plot(Y_test, '.b')
hold on
plot(y_hat_test, 'r')

legend('Valor real', 'Valor esperado')
xlabel('Tiempo')
ylabel('Salida')
hold off
%% c) Métricas de desempeño
% RMSE
error_val = mean((Y_val - y_hat_val).^2);
% FIT
fit_val = 1 - (error_val/var(Y_val));
% MAE 
mae_val = mean(abs(Y_val - y_hat_val));
disp(['   MSE val ', ' Fit val  ', 'MAE val'])
disp([error_val, fit_val, mae_val])
%% d) Parámetros de Intervalos Difusos - M. Covarianza
z= x_optim_ent;
y = Y_ent;
Npred = [1,8,16];
Nregs = size(z,2);
Nrules = size(model.a,1);
Ks = zeros(Nregs, Nregs, Nrules, length(Npred));
stds = zeros(Nrules, length(Npred));
% Estimamos K y std a p-pasos
for i=1:length(Npred)
    [~,z_pred] = ysimn(z, model, Npred(i));
    [Ks(:,:,:, i), stds(:, i)] = get_cov_params(z_pred,y(Npred(i):end), model);
end
%% Intervalo de incertidumbre para 1,8 y 16 pasos (Metodo de covarianza - A)
z = x_optim_test;
y = Y_test;
Npreds = [1,8,16];
alphas = zeros(length(Npreds), 9);
for i=1:length(Npreds)
    Npred = Npreds(i);
    % Evaluamos hasta la predicción
    [~,z_pred] = ysimn(z, model, Npred);
    for porcentaje=1:9
        alpha = 0;
        n_total = 0;
        [y_hat, I] = intervalos_cov(z_pred, model, stds(:,i), Ks(:,:,:,i));
        while n_total < porcentaje/10.0
            alpha = alpha + 0.01;
            n_total = calc_picp(y(Npred:end), y_hat - alpha*I, y_hat + alpha*I);
        end
        disp(n_total);
        alphas(i, porcentaje) = alpha;
    end
end
%% Graficar a 1,8,16 pasos + intervalos m.covarianza + validación
clc
z = x_optim_val;
y = Y_val;
Npreds = [1 8 16];
NNpreds = length(Npreds);
y_hats_ts = zeros(length(Y_val),NNpreds);
figure()
for idx=1:NNpreds
    Npred = Npreds(idx);
    [~,z_pred] = ysimn(z, model, Npred);
    subplot(NNpreds,1,idx);
    % Graficar intervalos de incertidumbre
    for porcentaje = flip(1:9)
        alpha = alphas(idx,porcentaje);
        [y_hat, I] = intervalos_cov(z_pred, model, stds(:,idx), Ks(:,:,:,idx));
        t = (1:length(y_hat)) + Npred;
        y_sup = y_hat + alpha*I;
        y_inf = y_hat - alpha*I;
        t2 = [t, fliplr(t)];
        inBetween = [y_sup; flipud(y_inf)];
        fill(t2, inBetween, [0.5 (1-porcentaje/10.0) 1], 'FaceAlpha', (10-porcentaje)/12.0);
        set(findobj(gca,'Type','Patch'),'EdgeColor', 'none'); % Quitar borde del fill
        hold on;
    end
    %guardar predicciones para mediciones
    if i == 1
        y_hats_ts(:,idx) = y_hat;
    else
        y_hats_ts(:,idx) = vertcat(zeros(Npred-1,1),y_hat);
    end
    % Graficar puntos reales
    plot(1:length(y), y(1:end),'b.', 'LineWidth', 0.3);
    hold on;
    % Graficar curva de estimación y_hat(k+i-1) (rojo oscuro)
    plot(t, y_hat, 'Color',[0.8 0 0] , 'LineWidth', 0.5);
    hold on;
    % Misma escala para todos los gráficos
    ylim([min(y), 1.5*max(y)]);
    xlim([0,500]); % Para visualizar mejor
    hold on;
    % Configuración de la gráfica
    xlabel('Tiempo'); 
    ylabel('Salida');
    title(sprintf('Modelo con intervalo de incertidumbre - Método de la covarianza - %d pasos', Npred));
    legend('90%','80%','70%', '60%','50%', '40%','30%','20%','10%',...
        'y_{val}', 'y_{hat}', 'Orientation','horizontal');
    hold off;
end
%% Métricas para predicciones a 8 y 18 Pasos

% RMSE
error_val_ts8 = mean((Y_val(9:end) - y_hats_ts(9:end,2)).^2);
% FIT
fit_val_ts8 = 1 - (error_val_ts8/var(Y_val(9:end)));
% MAE 
mae_val_ts8 = mean(abs(Y_val(9:end) - y_hats_ts(9:end,2)));

% RMSE
error_val_ts16 = mean((Y_val(17:end) - y_hats_ts(17:end,3)).^2);
% FIT
fit_val_ts16 = 1 - (error_val_ts16/var(Y_val(17:end)));
% MAE 
mae_val_ts16 = mean(abs(Y_val(17:end) - y_hats_ts(17:end,3)));

disp(['   MSE val ', ' Fit val  ', 'MAE val'])
disp([error_val_ts8, fit_val_ts8, mae_val_ts8])
disp([error_val_ts16, fit_val_ts16, mae_val_ts16])

%% jugando

Npred=10;

z_pred = x_optim_test;

for i = 1:Npred
      if i < Npred
         disp(size(z_pred))
         [y_pred, z_pred] = ysim3(z_pred, model);
         disp(size(z_pred))
      else
         [y_pred, ~] = ysim3(z_pred, model); %Ultimo: No usamos siguiente z_pred
      end
end
plot(y_pred, 'r-', 'LineWidth', 1);
hold on;
plot(Y_test, 'b.', 'LineWidth', 1);

%% Graficowos




