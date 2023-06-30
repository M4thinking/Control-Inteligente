%%
clear; clc; addpath("Toolbox TS NN/Toolbox difuso")
load('temperatura_10min.mat')


%% Creacion de la APRBS y la fijacion de temperatura como un timeSeries

Tfinal=length(temperatura');
Ts=1;
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


% corte solo para tomar los regresores pertenecient
[err_test, err_ent] = clusters_optimo(Y_test,Y_train,X_test(:,1:max_regs), X_train(:,1:max_regs), max_clusters);
figure()
plot(err_test, 'b')
hold on
plot(err_ent, 'r')
legend('Error de test', 'Error de entrenamiento')
title('Error en Función del Número de Reglas');
xlabel('Número de Reglas')
ylabel('Error Cuadrático Medio')

%% Optimizar modelo - Regresores
rules = 7; % Criterio anterior
[p, indices] = sensibilidad(Y_train, X_train(:,1:max_regs),rules); % rules = numero de clusters
n_regresores = 4; % Cambiar valor para mayor o menor número de regresores
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
xlabel('Tiempo')
ylabel('Salida')
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
Nregs = size(z,2)/2;
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
    disp(['   MSE val ', ' Fit val  ', 'MAE val'])
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

% creamos modelo discreto de la planta entegrada

HVAC=@(x,u,Ta,w,Ts) HVAC_dis(x,u,Ta,w,Ts);

x0 = [2;2]; %cuales serían las condiciones iniciales de esto?

% Tiempos de simulacion
Ts = 600; % Tiempo de muestreo
Tf = 12000; % Tiempo final 
% Loop de control
Ncontrol = Tf/Ts; % Número de pasos de control
Npred = 5; % Horizonde de prediccion

% Vectores para almacenar los resultados
x_vec = zeros(Ncontrol+1, 2); % Estados
y_vec = zeros(Ncontrol+1, 1); % Salida (theta)
u_vec = zeros(Ncontrol, 1);   % Entrada anterior

% Inicializamos el sistema
x_vec(1, :) = x0';
y_vec(1) = x0(1);
u_prev = 0;

% Creacion de referencias


t_ref = 0:Ts:Tf; % Vector de tiempo para la referencia a futuro
t_vec = 0:Ts:Tf; % Vector de tiempo para los resultados

ref1 = 20*ones(1,round(numel(t_ref)/3)); % Referencia 1 
ref2 = 18*ones(1,round(numel(t_ref)/3)); % Referencia 2 
ref3 = 25*ones(1,round(numel(t_ref)/3)); % Referencia 3 

ref=[ref1,ref2,ref3];

u0 = zeros(Npred, 1);  % Solución propuesta inicial
Ta=ysimn(x_optim_val(1:Ncontrol,:),model,Npred);
    
%noise
var=0.001;
w1=wgn(1,Ncontrol,var,'linear' );
w2=wgn(1,Ncontrol,var,'linear' );
w=[w1;w2];

for k = 1:Ncontrol-Npred
    
    % Ejecutar control predictivo
    next_ref = ref(k)*ones(Npred,1);
    [u_next, u0] = control_predictivo(Npred,HVAC,u0,x0,u_prev,next_ref,Ta(k),w(:,k),Ts);
    % Calcular el estado en el siguiente paso utilizando el modelo

    x_next = HVAC(x0,u_next,Ta(k),w(:,k),Ts);
    % Actualizar valores para el siguiente paso de control
    x0 = x_next;
    u_prev = u_next;
    x_vec(k+1, :) = x_next';
    y_vec(k+1) = x_next(1);
    u_vec(k) = u_next;

end

plots(t_vec, x_vec, y_vec, u_vec, ref, Ncontrol);



%% parte c)













