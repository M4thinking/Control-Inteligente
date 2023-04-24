clear; clc; addpath("Toolbox TS NN/Toolbox difuso")
%% a) Generación APRBS
aprbs = aprbsGen();
%% Correr simulink
out = sim('ident_model.slx');
%% b) Parametros modelo
max_regs = 5;
max_regs_list = 1:max_regs;
max_clusters = 16;

% Se cargan el vector Y de salida y la matriz X de regresores
porcentajes=[0.6,0.2,0.2];
[y ,x] = autoregresores(out.entrada,out.salida,max_regs);
[Y_val , Y_test, Y_ent, X_val, X_test, X_ent] = separar_datos(y, x, porcentajes);
%% Optimizar modelo - Reglas
[err_test, err_ent] = clusters_optimo(Y_test, Y_ent, X_test, X_ent, max_clusters);
rules = 2; % Criterio del codo
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')
title('Error en Función del Número de Reglas');
xlabel('Número de Reglas')
ylabel('Error Cuadrático Medio')
%% Optimizar modelo - Regresores
[p, indices] = sensibilidad(Y_ent, X_ent, rules); % rules = numero de clusters
n_regresores = 4; % Cambiar valor para mayor o menor número de regresores
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
%% Estimación de la salida
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
error_test = mean((Y_test - y_hat_test).^2);
% FIT
fit_test = 1 - (error_test/var(Y_test));
% MAE 
mae_ent = mean(abs(Y_test - y_hat_test));
disp(['   MSE test ', ' Fit test  ', 'MAE test'])
disp([error_test, fit_test, mae_ent])
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
%% Intervalo de incertidumbre para 1,8 y 16 pasos (Números difusos) (DEMORA MUCHO ~20min)
% pause
z = x_optim_test;
y = Y_test;
Nregs = size(z,2);
Nrules = size(model.a,1);
nu1 = 1; % Ponderador del PINAW
nu2 = 100; % Ponderador del PICP
Ns = Nregs*2*(Nrules+1);
Npreds = [1,8,16];
NNpreds = length(Npreds);
ss = zeros(Ns, NNpreds, 9);
for idx=1:NNpreds % Para cada predicción
    Npred = Npreds(idx);
    [~,z_pred] = ysimn(z, model, Npred);
    % Problema de optimización
    for porcentaje=flip(1:9) % Optimizamos para cada porcentaje
        % Reemplazamos fobj_fuzzy_nums con los valores conocidos hasta el momento
        J=@(s)fobj_fuzzy_nums(z_pred,model.a,model.b,model.g,s,y(Npred:end),nu1,nu2,1-porcentaje/10.0);
        % Particle Swarm Optimization y restricciones
        options = optimoptions('particleswarm','Display','iter', 'MaxIterations', 100);
        [sopt, ~] = particleswarm(J, Ns, zeros(Ns,1), ones(Ns,1), options);
        ss(:,idx, porcentaje) = sopt;
    end
end 

%% Guardar en archivo .mat
save('ss.mat', 'ss');
%% Resultado final en validación
load('sopt.mat', 'ss'); % Cargar optimo (evitar espera)
z = x_optim_val;
y = Y_val;
Nregs = size(z,2);
Nrules = 2;
nu1 = 1; % Ponderador del PINAW
nu2 = 100; % Ponderador del PICP
nu3 = 0; % Ponderador de la regulación L2 (Mejora -> PICP+ y PINAW-)
Ns = Nregs*2*(Nrules+1);
Npreds = [1,8,16];
NNpreds = length(Npreds);
figure()
for idx=1:NNpreds % Para cada predicción
    Npred = Npreds(idx);
    [~,z_pred] = ysimn(z, model, Npred);
    % Problema de optimización
    subplot(3,1,idx);
    for porcentaje=flip(1:9) % Optimizamos para cada porcentaje (al reves para el fill)
        [y_hat, y_sup, y_inf, PICP, PINAW, Jopt] = eval_fuzzy_nums(z_pred,model.a,model.b,model.g,ss(:,idx,porcentaje),y(Npred:end),nu1,nu2,1-porcentaje/10.0);
        % Consideramos el PICP y PINAW reales para comparar intervalos
        if porcentaje == 9
            disp([Npred, PICP, PINAW]);
        end
        t = (1:length(y_hat)) + Npred;
        t2 = [t, fliplr(t)];
        inBetween = [y_sup; flipud(y_inf)];
        fill(t2, inBetween, [0.5 (1-porcentaje/10.0) 1], 'FaceAlpha', (10-porcentaje)/12.0);
        hold on;
        set(findobj(gca,'Type','Patch'),'EdgeColor', 'none'); % Quitar borde del fill
        hold on;
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
    title(sprintf('Modelo con intervalo de incertidumbre - Números difusos - %d pasos', Npred));
    legend('90%','80%','70%', '60%','50%', '40%','30%','20%','10%',...
        'y_{val}', 'y_{hat}', 'Orientation','horizontal');
    hold off;
end

%% Comparación de intervalos de incertidumbre (métricas para cada predicción)

