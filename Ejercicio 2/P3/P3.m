clc;clear all
addpath('utils');
addpath('Toolbox TS NN/Toolbox difuso');
addpath('Toolbox TS NN/Toolbox NN');

%% Parte a) Definir modelo
model = @(u, x) pendcart(u, x);

% Condición inicial
x0 = [0; 0; pi-0.6; 0];  % [x, dx, theta, dtheta]

% Tiempos de simulacion
Ts = 0.1; % Tiempo de muestreo
Tf = 10; % Tiempo final en segundos

% Loop de control
Ncontrol = Tf/Ts; % Número de pasos de control
Npred = 7; % Horizonde de prediccion

% Vectores para almacenar los resultados
x_vec = zeros(Ncontrol+1, 4); % Estados
y_vec = zeros(Ncontrol+1, 1); % Salida (theta)
u_vec = zeros(Ncontrol, 1);   % Entrada anterior

% Inicializamos el sistema
x_vec(1, :) = x0';
y_vec(1) = x0(3);
u_prev = 0;

%% Parte b) Crear dos referencias
freq = 1; % Frecuenia sinusoide
t_vec = 0:Ts:Tf; % Vector de tiempo para los resultados
t_ref = 0:Ts:(Tf+(Npred+1)*Ts); % Vector de tiempo para la referencia a futuro
ref1 = pi*ones(1,numel(t_ref)); % Referencia 1 (constante = pi)
ref2 = pi+pi/4*cos(2*pi*freq*t_ref+pi/2); % Referencia 2 (pi + sinusoidal)

% Graficamos
figure(1)
plot(t_ref,ref1,'r',t_ref,ref2,'b')
legend('Referencia 1','Referencia 2')
xlim([0,Tf])
ylim([0,3*pi/2]);
xlabel('Tiempo [s]')
ylabel('Angulo [grados]')
title('Referencias')
grid on
hold off;

%% Parte c.0) Tuneo de parámetros (usado para efectos de implementación)
% freq = 1; % Frecuenia sinusoide
% Npred_tune = 4:1:12; % Horizonde de prediccion
% for i = 1:numel(Npred_tune)
%     Npred = Npred_tune(i);
%     t_ref = 0:Ts:(Tf+(Npred+1)*Ts); % Vector de tiempo para la referencia a futuro
%     ref1 = pi*ones(1,numel(t_ref)); % Referencia 1 (constante = pi)
%     ref2 = pi+pi/4*cos(2*pi*freq*t_ref+pi/2); % Referencia 2 (pi + sinusoidal)
%     theta_ref = ref2; % MODIFICAR REFERENCIA
%     u0 = zeros(Npred, 1);  % Solución propuesta inicial
%     for k = 1:Ncontrol
%         % Ejecutar control predictivo
%         next_ref = theta_ref(k+1:k+Npred)';
%         [u_next, u0] = control_predictivo(Npred,model,u0,x0,u_prev,next_ref);
%         % Calcular el estado en el siguiente paso utilizando el modelo
%         x_next = model(u_next, x0);
%         % Actualizar valores para el siguiente paso de control
%         x0 = x_next;
%         u_prev = u_next;
%         x_vec(k+1, :) = x_next';
%         y_vec(k+1) = x_next(3);
%         u_vec(k) = u_next;
%     end
%     % Graficar
%     plots(t_vec, x_vec, y_vec, u_vec, theta_ref, Ncontrol);
% end

%% Parte c) Controlador predictivo fenomenológico
Npred = 7; % Horizonde de prediccion
t_vec = 0:Ts:Tf; % Vector de tiempo para los resultados
t_ref = 0:Ts:(Tf+(Npred+1)*Ts); % Vector de tiempo para la referencia a futuro
ref1 = pi*ones(1,numel(t_ref)); % Referencia 1 (constante = pi)
ref2 = pi+pi/4*cos(2*pi*freq*t_ref+pi/2); % Referencia 2 (pi + sinusoidal)
theta_ref = ref2; % MODIFICAR REFERENCIA
u0 = ones(Npred, 1);  % Solución propuesta inicial
for k = 1:Ncontrol
    % Ejecutar control predictivo
    next_ref = theta_ref(k+1:k+Npred)';
    [u_next, u0] = control_predictivo(Npred,model,u0,x0,u_prev,next_ref);
    % Calcular el estado en el siguiente paso utilizando el modelo
    x_next = model(u_next, x0);
    % Actualizar valores para el siguiente paso de control
    x0 = x_next;
    u_prev = u_next;
    x_vec(k+1, :) = x_next';
    y_vec(k+1) = x_next(3);
    u_vec(k) = u_next;
end

%% Graficar
plot_states(t_vec, x_vec, u_vec, theta_ref, Ncontrol)

%% Generar gif
% m = 1; M = 5; L = 2;
% 
% t_vec_interp = 0:0.005:10;
% x_vec_interp = interp1(t_vec, x_vec, t_vec_interp, 'spline');% Interpolación con spline para mas fluidez en el gif
% plot_gif(t_vec_interp,x_vec_interp,m,M,L,'Control predictivo')

%% d) Controlador Predictivo Neuronal
% Utilizando los modelos identificados en el Ejercicio 1 o mejorados para el diseño de un
% controlador predictivo difuso o neuronal, que minimice la misma función de la parte c).
% Para esto:
% - Se aplica APRBS
% - Se identifica el modelo NN, de 4 estados, con 1 entrada y 1 salida
% - Se diseña un controlador predictivo neuronal, que minimice la misma función de la parte c).

%% Generar APRBS
x0 = [0; 0; pi; 0];  % [x, dx, theta, dtheta]
Tfinal = 6000; % Tiempo final [s]
Ts = 0.1; % Tiempo de muestreo
Tau = 8; % Tiempo de establecimiento [s]
Amin = -50; % Amplitud mínima
Amax = 55; % Amplitud máxima
t_vec = 0:Ts:Tfinal; % Vector de tiempo para los resultados
% Generar señal de entrada
aprbs = aprbsGen(Tfinal,Ts, Tau, Amin, Amax);
% Graficar APRBS
figure() 
plot(aprbs(:,1), aprbs(:,2), 'b-')
title('APRBS')
xlabel('Tiempo [s]')
ylabel('Amplitud [grados]')
grid on
hold off;
out = sim('pend.slx');
% Guardar datos en .mat (No guardar, ya se tienen)
% save('salida.mat','salida', 't_vec', 'Ts', 'aprbs');

% Cargar
load('salida.mat');
x=salida(:,1);
dx=salida(:,2);
theta=salida(:,3);
dtheta=salida(:,4);
% Graficar en mismo gráfico 4 subplots
figure()
subplot(4,1,1)
plot(t_vec,x) 
title('x')
subplot(4,1,2)
plot(t_vec,dx)
title('dx')
subplot(4,1,3)
% Graficar en grados
plot(t_vec,theta*180/pi)
title('theta')
subplot(4,1,4)
plot(t_vec,dtheta)
title('dtheta')
%%
% Entrenar red neuronal
% Se entrena la red neuronal con los datos de la simulación
% Se utiliza la función de entrenamiento trainlm
% Se utiliza la función de activación tangente hiperbólica
% Se utiliza la función de pérdida MSE
% Se utiliza el 70% entrenamiento, 20% validación, 10% test

nn_regs = 10;
% inputs son los regresores 60 regresores pasados para cada variable
x_regs = regresores(x, nn_regs);
dx_regs = regresores(dx, nn_regs);
theta_regs = regresores(theta, nn_regs);
dtheta_regs = regresores(dtheta, nn_regs);
inputs = [x_regs dx_regs theta_regs dtheta_regs]';
targets = [x(nn_regs+1:end) dx(nn_regs+1:end) theta(nn_regs+1:end) dtheta(nn_regs+1:end)]';
[Y_train, X_train, Y_val, X_val, Y_test, X_test] = split_data(targets, inputs, 0.7, 0.2, 0.1);
% Se calcula la media y desviación estándar de los datos de entrenamiento
states_mean = mean(X_train, 2);
states_std = std(X_train, 0, 2);
% Promedio y desviación estándar de salida
output_mean = mean(Y_train, 2);
output_std = std(Y_train, 0, 2);

% Se normalizan los datos de entrenamiento
X_train = (X_train - states_mean)./states_std;
Y_train = (Y_train - output_mean)./output_std;
% Se normalizan los datos de validación
X_val = (X_val - states_mean)./states_std;
Y_val = (Y_val - output_mean)./output_std;
% Se normalizan los datos de test
X_test = (X_test - states_mean)./states_std;
Y_test = (Y_test - output_mean)./output_std;

%% Cantidad de neuronas
Nnet = 100;
errores = zeros(Nnet,3);
for i=4:10
    disp(i)
    % Se crea la red neuronal con i capas
    mynet = fitnet([i 4]);
    % Se configura la función de entrenamiento
    mynet.trainFcn = 'trainscg';
    mynet.trainParam.showWindow=1; % Evita que se abra la ventana de entrenamiento
    % Se entrena la red neuronal
    mynet = train(mynet, X_train, Y_train, 'useParallel','yes');
    % Se simula la red neuronal
    Y_train_pred = mynet(X_train);
    Y_val_pred = mynet(X_val);
    % Se calcula el error de la red neuronal
    errores(i,1) = mse(Y_val - Y_val_pred);
    errores(i,2) =  mse(Y_train - Y_train_pred);
    errores(i,3) = i;
end

%% grafico error n neuronas
plot(errores(4:10,3),errores(4:10,1), 'r', 'LineWidth', 1.5); 
hold on
plot(errores(4:10,3),errores(4:10,2), 'b', 'LineWidth', 1.5);
title('Error asociado al número de neuronas')
legend('Error Validación', 'Error Entrenamiento')
xlabel('Número de Neuronas')
ylabel('Error')
hold off

%%
% Se crea la red neuronal con i capas
mynet = fitnet(6);
% Se configura la función de entrenamiento
mynet.trainFcn = 'trainscg';
mynet.trainParam.showWindow=1; % Evita que se abra la ventana de entrenamiento
% Se entrena la red neuronal
mynet = train(mynet, X_train, Y_train, 'useParallel','yes');
% Se simula la red neuronal
Y_train_pred = mynet(X_train);
Y_val_pred = mynet(X_val);
%% Medir sensibilidad de los regresores respecto a la salida
% Se calcula la sensibilidad de los regresores respecto al error de la red
sens = sensibilidad_regresores(mynet, X_val, Y_val_pred);
% Graficar sensibilidad de cada regresor respecto a cada variable (1 subplot por cada variable) (grafico de barras)
figure()
subplot(4,1,1)
bar(sens(:,1))
hold on
for i=nn_regs:nn_regs:length(sens(:,1))
    xline(i+0.5, 'LineWidth', 1.5);
end
hold off
title('Sensibilidad de los regresores respecto a x')
subplot(4,1,2)
bar(sens(:,2))
hold on
% Agregar linea vertical cada nn_regs regresores
for i=nn_regs:nn_regs:length(sens(:,2))
    xline(i+0.5, 'LineWidth', 1.5);
end
hold off
title('Sensibilidad de los regresores respecto a dx')
subplot(4,1,3)
bar(sens(:,3))
hold on
for i=nn_regs:nn_regs:length(sens(:,3))
    xline(i+0.5, 'LineWidth', 1.5);
end
hold off
title('Sensibilidad de los regresores respecto a theta')
subplot(4,1,4)
bar(sens(:,4))
hold on
for i=nn_regs:nn_regs:length(sens(:,4))
    xline(i+0.5, 'LineWidth', 1.5);
end
hold off
title('Sensibilidad de los regresores respecto a dtheta')
 

%% graficar sensibilidad de cada regresor respecto a cada variable (1 subplot por cada variable) (grafico de barras)
figure()
subplot(4,1,1)
bar(sens(:,1))
title('Sensibilidad de los regresores respecto a x')
subplot(4,1,2)
bar(sens(:,2))
title('Sensibilidad de los regresores respecto a dx') 
subplot(4,1,3)
bar(sens(:,3))
title('Sensibilidad de los regresores respecto a theta')
subplot(4,1,4)
bar(sens(:,4))
title('Sensibilidad de los regresores respecto a dtheta')

%%
% Se plotea la salida de la red neuronal en 4 subplots
figure()
subplot(4,1,1)
plot(Y_val(1,:)*output_std(1)+output_mean(1))
hold on
% Desnormalizar
plot(Y_val_pred(1,:)*output_std(1)+output_mean(1))
title('x')
legend('Real','Predicha') 
subplot(4,1,2)
plot(Y_val(2,:)*output_std(2)+output_mean(2))
hold on
plot(Y_val_pred(2,:)*output_std(2)+output_mean(2))
title('dx')
legend('Real','Predicha')
subplot(4,1,3)
plot(Y_val(3,:)*output_std(3)+output_mean(3))
hold on
% Graficar en grados
plot(Y_val_pred(3,:)*output_std(3)+output_mean(3))
title('theta')
legend('Real','Predicha')
subplot(4,1,4)
plot(Y_val(4,:)*output_std(4)+output_mean(4))
hold on
plot(Y_val_pred(4,:)*output_std(4)+output_mean(4))
title('dtheta')
legend('Real','Predicha')



%%
% Se crea la red neuronal
net = feedf;
% Se entrena la red neuronal
net = train(net,inputs,targets);
% Se simula la red neuronal
outputs = net(inputs);
% Se calcula el error de la red neuronal
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);
% Se plotea el error
figure, plotperform(tr)
% Se plotea la salida de la red neuronal
figure, plot(targets)
hold on
plot(outputs)


%% reinicializacion

% Condición inicial
x0 = [0; 0; pi-0.6; 0];  % [x, dx, theta, dtheta]

% Tiempos de simulacion
Ts = 0.1; % Tiempo de muestreo
Tf = 10; % Tiempo final en segundos

% Loop de control
Ncontrol = Tf/Ts; % Número de pasos de control
Npred = 7; % Horizonde de prediccion

% Vectores para almacenar los resultados
x_vec = zeros(Ncontrol+1, 4); % Estados
y_vec = zeros(Ncontrol+1, 1); % Salida (theta)
u_vec = zeros(Ncontrol, 1);   % Entrada anterior

% Inicializamos el sistema
x_vec(1, :) = x0';
y_vec(1) = x0(3);
u_prev = 0;
%% Parte f) Controlador predictivo fenomenológico con incertezas
theta_ref = ref1; % MODIFICAR REFERENCIA
u0 = ones(Npred, 1);  % Solución propuesta inicial
for k = 1:Ncontrol
    % Ejecutar control predictivo
    next_ref = theta_ref(k+1:k+Npred)';
    [u_next, u0] = control_predictivo(Npred,model,u0,x0,u_prev,next_ref);
    % Ruido gaussiano
    media = 0;         % Media del ruido
    desviacion = 0.1;  % Desviación estándar del ruido
    % Generar ruido blanco gaussiano
    ruido = desviacion*(randn(size(1))+media);
    u_next = u_next + ruido;
    % Calcular el estado en el siguiente paso utilizando el modelo + ruido
    x_next = model(u_next, x0);
    % Actualizar valores para el siguiente paso de control
    x0 = x_next;
    u_prev = u_next;
    x_vec(k+1, :) = x_next';
    y_vec(k+1) = x_next(3);
    u_vec(k) = u_next; % Guardamos lo que cree que aplicó
end

%% Graficar
plot_states(t_vec, x_vec, u_vec, theta_ref, Ncontrol)

%% Generar gif
% m = 1; M = 5; L = 2;
% 
% t_vec_interp = 0:0.015:10; % Aumentar la cantidad de frames
% % Interpolación con spline para mas fluidez en el gif
% x_vec_interp = interp1(t_vec, x_vec, t_vec_interp, 'spline');
% plot_gif(t_vec_interp,x_vec_interp,m,M,L,'Control predictivo con incerteza')


