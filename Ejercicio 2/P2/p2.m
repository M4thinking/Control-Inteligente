%%
clear; clc; addpath("Toolbox TS NN/Toolbox difuso")
load('temperatura_10min.mat')

%% Creacion de la APRBS y la fijacion de temperatura como un timeSeries

Tfinal=length(temperatura'); % muestra cada 10min
% se muestrea cada 10min
Ts=1; % 10min para el modelo de prediccion de temperatura
aprbs=aprbsGen(Tfinal,Ts);


t_simulacion=1:Ts:Tfinal;
temperatura=[t_simulacion;temperatura']'; %Supuse un espaciado de 1 s para la toma de muestras de T_a


%% Simulacion

sim('SimulinkHvac_2018a.slx')

%% parte a)

%Creacion de un modelo difuso para la variable T_a
max_regs =20;
max_regs_list = 1:max_regs;
max_clusters = 20;

% Se cargan el vector Y de salida y la matriz X de regresores
% Recordar que el orden de Y,X fue elegido arbitrariamente y su forma
% dependera de cada implementacion
% [y, x] = autoregresores(datos, max_regs);
% 
% [Y.val , Y.test, Y.ent, X.val, X.test, X.ent] = separar_datos(y, x, porcentajes);

porcentajes=[0.6,0.2,0.2];
entrada=zeros(length(temperatura(:,2)),1);
[Y,X]= autoregresores(entrada,temperatura(:,2),max_regs);
[Y_val, Y_test, Y_train, X_val, X_test, X_train] = separar_datos( Y,X, porcentajes);



%% Optimizar modelo - Reglas


% corte solo para tomar los regresores pertenecient (solo regs de y en X)
[err_test, err_ent] = clusters_optimo(Y_test,Y_train,X_test(:,1:max_regs), X_train(:,1:max_regs), max_clusters);
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')
title('Error en Función del Número de Reglas');
xlabel('Número de Reglas')
ylabel('Error Cuadrático Medio')
hold off

%% Optimizar modelo - Regresores
rules = 2; % Criterio anterior
[p, indices] = sensibilidad(Y_train, X_train(:,1:max_regs),rules); % rules = numero de clusters
n_regresores = 3; % Cambiar valor para mayor o menor número de regresores
best_indices = [];
for i=1:n_regresores % Descartamos peor regresor
    [~, idx] = max(indices);
    best_indices = [best_indices, idx];
    indices(idx) = 0;
end

x_optim_ent = X_train(:, sort(best_indices, 'ascend'));
x_optim_test = X_test(:, sort(best_indices, 'ascend'));
x_optim_val = X_val(:, sort(best_indices, 'ascend'));

%% Entrenar modelo
[model, ~] = TakagiSugeno(Y_train, x_optim_ent, rules, [1 2 1]);

%% Predicciones
y_hat_ent = ysim(x_optim_ent, model.a, model.b, model.g);
y_hat_test = ysim(x_optim_test, model.a, model.b, model.g);
y_hat_val = ysim(x_optim_val, model.a, model.b, model.g);

figure()
plot(Y_val, '.b')
hold on
plot(y_hat_val, 'r')

legend('Valor real', 'Valor esperado')
xlabel('Tiempo [min]')
ylabel('Salida T°')
hold off


%% Predicciones a 1,5 y 10 pasos sobre conjunto de test
clc
% predict = x_optim_ent;
% net_optim_structure = my_ann_exporter(net_optim);
% y_hat_ent = my_ann_evaluation(net_optim_structure, x_optim_ent');
z = x_optim_test;
y = Y_test;
Npreds = [1,5, 10];
NNpreds = length(Npreds);
%y_hat_ent = my_ann_evaluation(net_optim_structure, predict');
%size(predict)
figure()
for i=1:NNpreds
    
    z = x_optim_test;
    Npred = Npreds(i);
    [y_hat, ~] = ysimn(z, model, Npred);
    
    % Métricas relevantes
    disp(['Predicciones a ', num2str(Npred), 'pasos.'])
    % RMSE
    error_test_nn = mean((y(Npred:end) - y_hat).^2);
    % FIT
    fit_test_nn = 1 - (error_test_nn/var(y(Npred:end)));
    % MAE 
    mae_test_nn = mean(abs(y(Npred:end) - y_hat));
    disp(['   MSE Test ', ' Fit Test  ', 'MAE Test'])
    disp([error_test_nn, fit_test_nn, mae_test_nn])
    subplot(NNpreds,1,i)
    plot((1:length(y_hat)), y_hat, 'r-')
    hold on
    plot(y, '.b')
    hold on
    xlim([0,1000]); % Para visualizar mejor
    hold on;
    % Misma escala para todos los gráficos
    xlim([0,1000]); % Para visualizar mejor
    title(['Predicción en entrenamiento - Modelo Difuso - ', num2str(Npred), ' pasos'])
    xlabel('Tiempo')
    ylabel('Salida')
    legend('Valor esperado', 'Valor real')
end
hold off


%% Seccion b)
a = model.a;
b = model.b;
g = model.g;
x0 = [2;2]; %cuales serían las condiciones iniciales de esto?

% Tiempos de simulacion
% Tiempo de muestreo de 10 minutos
Ts = 10*60; % [s]
Tf = 3*60*60; % [s] 3 horas
N = Tf/Ts+1; % Número de pasos de simulación

t_ref = 0:Ts:Tf; % Vector de tiempo para la referencia a futuro
t_vec = 0:Ts:Tf; % Vector de tiempo para los resultados

ref1 = 20*ones(1,1+round(numel(t_ref)/3)); % Referencia 1
ref2 = 18*ones(1,round(numel(t_ref)/3)); % Referencia 2
ref3 = 25*ones(1,round(numel(t_ref)/3)); % Referencia 3
refs = [ref1 ref2 ref3];
% crear objeto signal 
ref = timeseries(refs, t_ref);

% Generar señal de temperatura ambiente
% Señal de temperatura ambiente

Tamb = timeseries(temperatura(1:N, 2), t_vec);

% Enviar a simulink como una señal (from workspace)

%%
% Aplicar simulacion de Hvac_control.slx

% Add model to simulink
open_system('Hvac_control.slx');
% Tiempos
% Simular en tiempo continuo
set_param('Hvac_control','Solver','ode45');
set_param('Hvac_control','StartTime','0');
set_param('Hvac_control','StopTime','Tf');
set_param('Hvac_control/From Workspace','SampleTime','Ts');
% Run simulation
out = sim('Hvac_control.slx');
%%
ref_sim = squeeze(ref.Data);
u_sim = U;
y_sim = T1;
t_sim = ref.Time;

% Graficar resultados , subplot 1: ref + y, subplot 2: u
figure()
subplot(2,1,1)
% plot(t_ref, ref_sim, 'r.-')
stairs(t_ref, ref_sim, 'r-')
hold on
plot(t_sim, y_sim, 'b-')
legend('Referencia', 'Salida')
xlabel('Tiempo [s]')
ylabel('Temperatura [°C]')
title('Simulación de sistema HVAC')
hold off
subplot(2,1,2)
% formato stairs
stairs(t_sim, u_sim, 'b-')
xlabel('Tiempo [s]')
ylabel('Flujo másico')
title('Flujo en HVAC')
hold off

%%
plots(t_vec, x_vec, y_vec, u_vec, ref, Ncontrol);



%% parte c)













