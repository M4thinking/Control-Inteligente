clc;clear all
addpath('utils');
addpath('Toolbox TS NN/Toolbox difuso');
addpath('Toolbox TS NN/Toolbox NN');
global model mem

mem = zeros(3, 60);
%% Parte a) Definir modelo
model = @(u, x) pendcart(u, x);

% Condición inicial
x0 = [0; 0; 0; 0];  % [x, dx, theta, dtheta]

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
% Condición inicial
x0 = [0; 0; 0; 0];  % [x, dx, theta, dtheta]
Npred = 7; % Horizonde de prediccion
t_vec = 0:Ts:Tf; % Vector de tiempo para los resultados
t_ref = 0:Ts:(Tf+(Npred+1)*Ts); % Vector de tiempo para la referencia a futuro
ref1 = pi*ones(1,numel(t_ref)); % Referencia 1 (constante = pi)
ref2 = pi+pi/4*cos(2*pi*freq*t_ref+pi/2); % Referencia 2 (pi + sinusoidal)
theta_ref = ref1; % MODIFICAR REFERENCIA
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

%% Generar datos para trayectoria conocida
%% Generar APRBS
load('salida.mat');
figure() 
plot(aprbs(:,1), aprbs(:,2), 'b-')
title('APRBS')
xlabel('Tiempo [s]')
ylabel('Amplitud [grados]')
grid on
hold off;

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
nn_regs = 60; % inputs son los nn_regs regresores pasados por variable
u = salida(:,3);

theta=salida(:,2);
porcentajes=[0.6,0.2,0.2];
% Modelo NARX para angulo
[y ,x]=autoregresores(u,theta,nn_regs);
[Y_val_1 , Y_test_1, Y_ent_1, X_val_1, X_test_1, X_ent_1] = separar_datos(y, x, porcentajes);


x=salida(:,1);
% Modelo NARX para posicion
[y ,x]=autoregresores(u,x,nn_regs);
[Y_val_2 , Y_test_2, Y_ent_2, X_val_2, X_test_2, X_ent_2] = separar_datos(y, x, porcentajes);

%% Predicciones con entrada conocida
load('net_theta.mat', 'net_theta')
Y_pred_1 = my_ann_evaluation(net_theta, X_test_1');
figure()
plot(Y_test_1*180/pi, 'b.', 'LineWidth', 1)
hold on
plot(Y_pred_1*180/pi, 'r-')
legend('Valor real', 'Valor esperado')
hold off


load('net_pos.mat', 'net_pos')
Y_pred_2 = my_ann_evaluation(net_pos, X_test_2');
figure()
plot(Y_test_2, 'b.', 'LineWidth', 1)
hold on
plot(Y_pred_2, 'r-')
legend('Valor real', 'Valor esperado')
hold off

%% 
% Add model to simulink
open_system('sistema_controlado.slx');
x0 = [0; 0; pi; 0];  % [x, dx, theta, dtheta]
Tf = 100; % Tiempo final en segundos
Npred = 7; % Horizonde de prediccion
t_vec = 0:Ts:Tf; % Vector de tiempo para los resultados
t_ref = 0:Ts:(Tf+(Npred+1)*Ts); % Vector de tiempo para la referencia a futuro
ref1 = pi*ones(1,numel(t_ref)); % Referencia 1 (constante = pi)
ref2 = pi+pi/4*cos(2*pi*freq*t_ref+pi/2); % Referencia 2 (pi + sinusoidal)
r = ref1;

% Tiempos
% Simular en tiempo continuo
set_param('sistema_controlado','Solver','ode45');
set_param('sistema_controlado','StartTime','0');
set_param('sistema_controlado','StopTime','Tf');
% Run simulation
out = sim('sistema_controlado.slx');
%%
% Graficar out.y (theta), out.x (posicion), out.u (control)
figure()
subplot(3,1,1)
plot(out.tout, out.y*180/pi, 'b-', 'LineWidth', 1)
legend('Valor real', 'Valor esperado')
title('Theta')
subplot(3,1,2)
plot(out.tout, out.x(:,1), 'b-', 'LineWidth', 1)
subplot(3,1,3)
plot(out.tout(6:end), out.u, 'b-', 'LineWidth', 1)
title('Accion de control')
hold off

%% Predicciones a N pasos
z_pred = X_test_1;
y = Y_test_1;
net_optim_structure = net_theta;
Npred = 7;
Nregs = 60;
for j=1:Npred
    y_hat = my_ann_evaluation(net_optim_structure, z_pred');
    if j < Npred
        z_pred = [y_hat(1:end-1)', z_pred(1:end-1, 2:Nregs), z_pred(2:end,Nregs+1:end)];
    end
end

figure()
plot(((Npred+1):length(Y_test_1))-Npred, Y_test_1(Npred+1:end)*180/pi, 'b.', 'LineWidth', 1)
hold on
plot(y_hat*180/pi, 'r-')
legend('Valor real', 'Valor esperado')
hold off

nu1 = 1; % Ponderador del PINAW
nu2 = 100; % Ponderador del PICP
nu3 = 0; % Ponderador de la regulación L2 (Mejora -> PICP+ y PINAW-)
Nneuronas = 15;
Ns = 2*(Nneuronas+1);
ss = zeros(Ns, 9);
% Problema de optimización
z_pred = X_test_1;
for j=1:Npred
    y_hat = my_ann_evaluation(net_optim_structure, z_pred');
    if j < Npred
        z_pred = [y_hat(1:end-1)', z_pred(1:end-1, 2:Nregs), z_pred(2:end,Nregs+1:end)];
    end
end
% Problema de optimización
porcentaje=9; % Optimizamos para cada porcentaje
% Reemplazamos fobj_fuzzy_nums con los valores conocidos hasta el momento
J=@(s)f_obj_fuzzy_nums_nn(z_pred,net_optim_structure,s,y(Npred:end),nu1,nu2,nu3,1-porcentaje/10.0);
% Particle Swarm Optimization y restricciones
options = optimoptions('particleswarm','Display','iter', 'MaxIterations', 100);
[sopt, ~] = particleswarm(J, Ns, zeros(Ns,1), ones(Ns,1), options);
ss(:, porcentaje) = sopt;

% Graficar
figure()
porcentaje=9; % Optimizamos para cada porcentaje (al reves para el fill)
[y_hat, y_sup, y_inf, PICP, PINAW, Jopt] = eval_fuzzy_nums_nn(z_pred,net_optim_structure,ss(:,porcentaje),y(Npred:end),nu1,nu2,nu3,1-porcentaje/10.0);
% Consideramos el PICP y PINAW reales para comparar intervalos
if porcentaje == 9
    disp([Npred, PICP, PINAW]);
end
t = (1:length(y_hat));
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

% Misma escala para todos los gráficos
xlim([0,1000]); % Para visualizar mejor
hold on;
% Configuración de la gráfica
xlabel('Tiempo'); 
ylabel('Salida');
title(sprintf('Modelo con intervalo de incertidumbre - Números difusos - %d pasos', Npred));
legend('90%',...
    'y_{val}', 'y_{hat}', 'Orientation','horizontal');
hold off;


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


