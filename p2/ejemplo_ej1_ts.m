clear
clc

addpath("Toolbox TS NN/Toolbox difuso")
%% Generación APRBS
aprbs = aprbsGen();
%% Correr simulink
out = sim('ident_model.slx');
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
%% Parámetros de Intervalos Difusos - M. Covarianza
z = x_optim_ent;
% [y, [h1,...,hj]] con j = número de reglas
[y, h] = ysim2(z, model.a, model.b, model.g); % y: salida, h: grados de activación normalizado
% Obtenemos lso Phi (P), Pi,j = h(i,j)*x(i,:)^T
[lenz, regs] = size(z);
K = zeros(regs, regs, rules);
for j=1:rules
    Pj = zeros(regs, length(z));
    for i=1:lenz
        Pij = h(i,j)*z(i,:)';
        Pj(:,i) = Pij;
    end
    K(:,:,j) = (Pj*Pj')^-1;
end
% Calculamos el error ej = yj - y_hatj (yj = y*hj)
e_ent = zeros(lenz, rules);
std_ent = zeros(rules,1);
y_hatj = y_hat_ent;
for j=1:rules
    y_hatj = y.*h(:,j);
    yj = Y_ent.*h(:,j);
    e_ent(:,j) = yj - y_hatj;
    std_ent(j) = std(e_ent(:,j));
end

%% Intervalo de incertidumbre para cada regla en entrenamiento
alphas = []; % Alphas se entrenan en test
for porcentaje=1:9
    alpha = 0;
    n_total = 0;
    [y_hat, I] = intervalos_cov(x_optim_test, model.a, model.b, model.g, std_ent, K);
    while n_total < porcentaje/10.0
        y_sup = y_hat + alpha*I;
        y_inf = y_hat - alpha*I;
        alpha = alpha + 0.01;
        n_total = sum(y_inf<=Y_test & Y_test<=y_sup)/double(sum(length(Y_test)));
    end
    disp(n_total);
    alphas = [alphas, alpha];
end
%---------------------------------
%% Validacion (Final)
y_sups = []; % Limite superior intervalo
y_infs = []; % Limite inferior intervalo
for i=1:length(alphas)
    [y_hat, I] = intervalos_cov(x_optim_val, model.a, model.b, model.g, std_ent, K);
    y_sup = y_hat + alphas(i)*I;
    y_inf = y_hat - alphas(i)*I;
    y_sups = [y_sups, y_sup];
    y_infs = [y_infs, y_inf];
end

len = length(Y_val); % Cambiar a length(Y_ent) para ver todos los datos.
t = 1:len;
figure();
% Graficar intervalos de incertidumbre
for i = flip(1:9)
    alpha = alphas(i);
    y_sup = y_sups(1:len,i);
    y_inf = y_infs(1:len,i);
    t2 = [t, fliplr(t)];
    inBetween = [y_sup; flipud(y_inf)];
    fill(t2, inBetween, [0.5 (1-i/10.0) 1], 'FaceAlpha', (10-i)/12.0);
    hold on;
end
% Graficar curva de estimación y_hat_val
plot(t, y_hat(1:len), 'r-', 'LineWidth', 1);
hold on;
% Graficar puntos Y_val
scatter(t, Y_val(1:len), 5, 'b', 'filled');
hold on;
% Configuración de la gráfica
xlabel('Tiempo');
ylabel('Salida');
title('Modelo con intervalo de incertidumbre - Método de la covarianza');
legend('90%','80%','70%', '60%','50%', '40%','30%','20%','10%',...
    'Estimación', 'Datos de entrenamiento');

%% Evaluar predicciones a 8 y 16 pasos para sintonizar alpha (Metodo A)
n_pred = 16;
regs = 2;
z = x_optim_test;
y = Y_test;
[yk, I_pred] = eval_pred(z,y, model, std_ent, K, regs, n_pred);
%% Intervalo de incertidumbre para 1,8 y 16 pasos
alphas = zeros(3,9); % Alphas se entrenan en test

preds = [1,8,16];
nof_preds = length(preds);
for idx=1:nof_preds
    pred = preds(idx);
    for porcentaje=1:9
        alpha = 0;
        n_total = 0;
        while n_total < porcentaje/10.0
            y_sup = yk(:, pred) + alpha*I_pred(:, pred);
            y_inf = yk(:, pred) - alpha*I_pred(:, pred);
            alpha = alpha + 0.01;
            n_total = sum(y_inf(1:end-pred)<=y(pred+1:end) & y(pred+1:end)<=y_sup(1:end-pred))/double(sum(length(y)));
        end
        disp(porcentaje);
        alphas(idx, porcentaje) = alpha;
    end
end 
 
%%
% Graficar predicciones a 1,8,16 pasos con intervalos de incertidumbre + datos de validación
preds = [1,8,16];
colors = ['r', 'g', 'b'];
nof_preds = length(preds);
z = x_optim_val;
y = Y_val;
[yk, I_pred] = eval_pred(z,y, model, std_ent, K, regs, n_pred);
figure()
for idx=1:nof_preds
    disp(1);
    pred = preds(idx);
    subplot(nof_preds,1,idx);
    plot(1,1);
    % Graficar intervalos de incertidumbre
    t = 1:length(yk)-pred;
    for i = flip(1:9)
        alpha = alphas(idx,i);
        y_sup = yk(1:end-pred, pred) + alpha*I_pred(1:end-pred, pred);
        y_inf = yk(1:end-pred, pred) - alpha*I_pred(1:end-pred, pred);
        t2 = [t, fliplr(t)];
        inBetween = [y_sup; flipud(y_inf)];
        fill(t2, inBetween, [0.5 (1-i/10.0) 1], 'FaceAlpha', (10-i)/12.0);
        % Quitar borde del fill
        set(findobj(gca,'Type','Patch'),'EdgeColor', 'none');
        hold on;
    end
    
    % Graficar puntos reales
    plot(pred+1:length(y), y(1:end-pred),'b.', 'LineWidth', 0.3);
    
    % Graficar curva de estimación y_hat(k+i-1) (rojo oscuro)
    plot(t, yk(1:end-pred, pred), 'Color',[0.8 0 0] , 'LineWidth', 0.5);
    hold on;
    
    % Misma escala para todos los gráficos
    limy = 5*max(abs(y));
    ylim([-limy, limy]);
    hold on;
    % Configuración de la gráfica
    xlabel('Tiempo'); 
    ylabel('Salida');
    title(sprintf('Modelo con intervalo de incertidumbre - Método de la covarianza - %d pasos', pred));
    legend('90%','80%','70%', '60%','50%', '40%','30%','20%','10%',...
        'y_{val}', 'y_{hat}', 'Orientation','horizontal');
end
