clear; clc; 
%%
addpath("Toolbox TS NN/Toolbox NN")
%% Generación APRBS

y0 = [0; 0; pi-0.1; 0];
Tfinal=1200;
Ts = 0.1;
aprbs = aprbsGen(Tfinal, Ts);
%% Correr simulink
out = sim('ident_pendcart.slx');
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
% ylabel('Grados [°C]')
%% Parametros modelo
max_regs = 60;
max_regs_list = 1:max_regs;
max_hlayer = 5:5:30;


% Se cargan el vector Y de salida y la matriz X de regresores
% Recordar que el orden de Y,X fue elegido arbitrariamente y su forma
% dependera de cada implementacion

% [y, x] = autoregresores(datos, max_regs);
% 
% [Y.val , Y.test, Y.ent, X.val, X.test, X.ent] = separar_datos(y, x, porcentajes);
% load("autorregresores.mat")

x=salida(:,1);
dx=salida(:,2);
theta=salida(:,3);
dtheta=salida(:,4);
plot(sin(theta));
porcentajes=[0.6,0.2,0.2];
[y ,x]=autoregresores(entrada,theta,max_regs);
[Y_val , Y_test, Y_ent, X_val, X_test, X_ent] = separar_datos(y, x, porcentajes);


%% Optimizacion numero de neuronas
Nnet = 60;
errores = zeros(Nnet,3);
for i=1:Nnet
    disp(i)
    net_ent = fitnet(i); % 15 neuronas en capa oculta
    net_ent.trainFcn = 'trainscg'; % Funcion de entrenamiento
    net_ent.trainParam.showWindow=1; % Evita que se abra la ventana de entrenamiento
    net_ent = train(net_ent,X_ent',Y_ent', 'useParallel','yes');
    y_p_ent = net_ent(X_ent')'; % Se genera una prediccion en conjunto de entrenamiento
    y_p_test = net_ent(X_test')'; % Se genera una prediccion en conjunto de test
    errtest= (sqrt(sum((y_p_test-Y_test).^2)))/length(Y_test); % Se guarda el error de test
    errent= (sqrt(sum((y_p_ent-Y_ent).^2)))/length(Y_ent); % Se guarda el error de entrenamiento
    errores(i,1) = errtest;
    errores(i,2) = errent;
    errores(i,3) = i;
    
    
end
%% grafico error n neuronas
plot(errores(:,3),errores(:,1), 'r', 'LineWidth', 1.5); 
hold on
plot(errores(:,3),errores(:,2), 'b', 'LineWidth', 1.5);
title('Error asociado al número de neuronas')
legend('Error Test', 'Error Entrenamiento')
xlabel('Número de Neuronas')
ylabel('Error')
hold off

%% entrenamiento red para optimizar
Nopt = 2;
net_ent = fitnet(Nopt);
net_ent.trainFcn = 'trainscg';  
net_ent.trainParam.showWindow=0;
net_ent = train(net_ent,X_ent',Y_ent', 'useParallel','yes');
%% eleccion regresores por sensibilidad
[p, indices] = sensibilidad_nn(X_ent, net_ent); % rules = numero de clusters
n_regresores = 120; % Valen similar, asi que las tomaremos todas
best_indices = [];
for i=1:n_regresores % Descartamos peor regresor
    [~, idx] = max(indices);
    best_indices = [best_indices, idx];
    indices(idx) = 0;
    %Aqui puedes dropear alguno y volver a entrenar si quieres
end

x_optim_ent = X_ent(:, sort(best_indices, 'ascend'));
x_optim_test = X_test(:, sort(best_indices, 'ascend'));
x_optim_val = X_val(:, sort(best_indices, 'ascend'));
%% entrenamiento red optima
net_optim = fitnet(4);
net_optim.trainFcn = 'trainscg';  
net_optim.trainParam.showWindow=0;
net_optim = train(net_optim,x_optim_ent',Y_ent', 'useParallel','yes');
%% Guardar la red
my_net = my_ann_exporter(net_optim);
save('my_net.mat', 'my_net');
%% Predicciones
load('my_net_p3.mat', 'my_net')
net_optim_structure = my_net;
y_hat = my_ann_evaluation(net_optim_structure, x_optim_ent');
%y_hat = net_optim(x_optim_val')';

figure()
plot(Y_ent, 'b.', 'LineWidth', 1)
hold on
plot(y_hat, 'r-')

legend('Valor real', 'Valor esperado')

%% Predicciones
y_hat_test = my_ann_evaluation(net_optim_structure, x_optim_test');
%y_hat = net_optim(x_optim_val')';

figure()
plot(Y_test, '.b')
hold on
plot(y_hat_test, 'r')
title('Predicción en test - Modelo Neuronal')
xlabel('Tiempo')
ylabel('Salida')
legend('Valor real', 'Valor esperado')

%% Métricas de desempeño
% RMSE
error_test_nn = mean((Y_test - y_hat_test').^2);
% FIT
fit_test_nn = 1 - (error_test_nn/var(Y_test));
% MAE 
mae_test_nn = mean(abs(Y_test - y_hat_test'));

disp(['   MSE test ', ' Fit test  ', 'MAE test'])
disp([error_test_nn, fit_test_nn, mae_test_nn])

%% Predicciones a 1, 8 y 16 pasos
clc
% predict = x_optim_ent;
% net_optim_structure = my_ann_exporter(net_optim);
% y_hat_ent = my_ann_evaluation(net_optim_structure, x_optim_ent');

Npreds = [1, 30, 60];
NNpreds = length(Npreds);
y = Y_ent;
z = x_optim_ent;
Nregs = size(z,2)/2;
%y_hat_ent = my_ann_evaluation(net_optim_structure, predict');
%size(predict)
figure()
for i=1:NNpreds
    Npred = Npreds(i);
    z = x_optim_ent;
    for j=1:Npred
        disp(size(z))
        y_hat = my_ann_evaluation(net_optim_structure, z');
        % z = [yk-1, yk-2, uk-1, uk-2]
        z = [y_hat(1:end-1)', z(1:end-1, 2:Nregs), z(2:end,Nregs+1:end)];
    end
    subplot(NNpreds,1,i)
    plot(y, 'b.')
    hold on
    plot(1:length(y_hat), y_hat, 'r-')
    ylim([min(y), 1.5*max(y)]);
    xlim([0,1000]); % Para visualizar mejor
    title(['Predicción en entrenamiento - Modelo Neuronal - ', num2str(Npred), ' pasos'])
    xlabel('Tiempo')
    ylabel('Salida')
    legend('Valor esperado', 'Valor real')
end
hold off

%% Predicciones a 1,30 y 60 pasos sobre conjunto de test
clc
% predict = x_optim_ent;
% net_optim_structure = my_ann_exporter(net_optim);
% y_hat_ent = my_ann_evaluation(net_optim_structure, x_optim_ent');
y = Y_test;
Npreds = [1, 30, 60];
NNpreds = length(Npreds);
Nregs = size(z,2)/2;
%y_hat_ent = my_ann_evaluation(net_optim_structure, predict');
%size(predict)
figure()
for i=1:NNpreds
    z = x_optim_test;
    Npred = Npreds(i);
    for j=1:Npred
        y_hat = my_ann_evaluation(net_optim_structure, z');
        z = [y_hat(1:end-1)', z(1:end-1, 2:Nregs), z(2:end,Nregs+1:end)];
    end

    y_hat = y_hat(1:end)';
    % Métricas relevantes
    disp(['Predicciones a ', num2str(Npred), 'pasos.'])
    % RMSE
    error_test_nn = mean((y(Npred:end) - y_hat).^2);
    % FIT
    fit_test_nn = 1 - (error_test_nn/var(y(Npred:end)));
    % MAE 
    mae_test_nn = mean(abs(y(Npred:end) - y_hat));
    disp(['   MSE val ', ' Fit val  ', 'MAE val'])
    disp([error_test_nn, fit_test_nn, mae_test_nn])
    subplot(NNpreds,1,i)
    plot((1:length(y_hat)), y_hat, 'r-')
    hold on
    plot(y, '.b')
    hold on
    % Misma escala para todos los gráficos
    xlim([0,1000]); % Para visualizar mejor
    title(['Predicción en entrenamiento - Modelo Neuronal - ', num2str(Npred), ' pasos'])
    xlabel('Tiempo')
    ylabel('Salida')
    legend('Valor esperado', 'Valor real')
end
hold off

%% Interalos - Numeros difusos - Predicciones a 1, 8 y 18 Pasos
clc
z = x_optim_test;
y = Y_test;
Nregs = size(z,2);
Nneuronas = 4;
nu1 = 1; % Ponderador del PINAW
nu2 = 100; % Ponderador del PICP
nu3 = 0; % Ponderador de la regulación L2 (Mejora -> PICP+ y PINAW-)
Ns = 2*(Nregs+1)*Nneuronas;
Npreds = [1,8,16];
NNpreds = length(Npreds);
ss = zeros(Ns, NNpreds, 9);
for idx=1:NNpreds % Para cada predicción
    Npred = Npreds(idx);
    z_pred = z;
    for j=1:Npred
        y_hat = my_ann_evaluation(net_optim_structure, z_pred');
        if j < Npred
            z_pred = [y_hat(1:end-1)', z(1:end-1, 2:Nregs), z(2:end,Nregs+1:end)];
        end
    end
    % Problema de optimización
    for porcentaje=flip(1:9) % Optimizamos para cada porcentaje
        % Reemplazamos fobj_fuzzy_nums con los valores conocidos hasta el momento
        J=@(s)f_obj_fuzzy_nums_nn(z_pred,net_optim_structure,s,y(Npred:end),nu1,nu2,1-porcentaje/10.0);
        % Particle Swarm Optimization y restricciones
        options = optimoptions('particleswarm','Display','iter', 'MaxIterations', 100);
        [sopt, ~] = particleswarm(J, Ns, zeros(Ns,1), ones(Ns,1), options);
        ss(:,idx, porcentaje) = sopt;
    end
end 

%% Resultado final en validación
load('sopt.mat', 'ss'); % Cargar optimo (evitar espera)
z = x_optim_val;
y = Y_val;
Nregs = size(z,2);
Nneuronas = 4;
nu1 = 1; % Ponderador del PINAW
nu2 = 100; % Ponderador del PICP
nu3 = 0; % Ponderador de la regulación L2 (Mejora -> PICP+ y PINAW-)
Ns = 2*(Nregs+1)*Nneuronas;
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


%% 
clc
z = x_optim_test;
y = Y_test;
porcentaje = 9;
nu1 = 1000000; % Ponderador del PINAW
nu2 = 1; % Ponderador del PICP
nu3 = 2;
Nneuronas = 4;
Ns = 2*(Nregs+1)*Nneuronas;
s = zeros(Ns,1);
f_obj_fuzzy_nums_nn(z,net_optim_structure,s,y(1:end),nu1,nu2,nu3,1-porcentaje/10.0);
%%
% Reemplazamos fobj_fuzzy_nums con los valores conocidos hasta el momento
J=@(s)f_obj_fuzzy_nums_nn(z_pred,net_optim_structure,s,y,nu1,nu2,nu3,0.1);
% Particle Swarm Optimization y restricciones
options = optimoptions('particleswarm','Display','iter', 'MaxIterations', 100);
[sopt, ~] = particleswarm(J, Ns, zeros(Ns,1), ones(Ns,1), options);
ss1 = sopt;
%%
[~, ~, ~, PICP, PINAW, ~] = eval_fuzzy_nums_nn(z,net_optim_structure,ss1,y,nu1,nu2,nu3,1-porcentaje/10.0);
disp([PICP, PINAW]);
%%
z = x_optim_val;
y = Y_val;
[y_hat, y_sup, y_inf, PICP, PINAW, Jopt] = eval_fuzzy_nums_nn(z,net_optim_structure,ss1,y(1:end),nu1,nu2,1-porcentaje/10.0);
% Consideramos el PICP y PINAW reales para comparar intervalos
if porcentaje == 9
    disp([Npred, PICP, PINAW]);
end
t = (1:length(y_hat)) + 1;
t2 = [t, fliplr(t)];
inBetween = [y_sup; flipud(y_inf)];
fill(t2, inBetween, [0.5 (1-porcentaje/10.0) 1], 'FaceAlpha', (10-porcentaje)/12.0);
hold on;
set(findobj(gca,'Type','Patch'),'EdgeColor', 'none'); % Quitar borde del fill
hold on;
% Graficar puntos reales
plot(1:length(y), y(1:end),'b.', 'LineWidth', 0.3);
hold on;
% Graficar curva de estimación y_hat(k+i-1) (rojo oscuro)
plot(t, y_hat, 'Color',[0.8 0 0] , 'LineWidth', 0.5);
hold on;



