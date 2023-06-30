clc;clear all
%% Definición de parámetros
addpath('utils');

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
t_ref = 0:Ts:(Tf+Npred*Ts); % Vector de tiempo para la referencia a futuro
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

%% Parte c.0) Tuneo de parámetros
freq = 1; % Frecuenia sinusoide
Npred_tune = 4:1:12; % Horizonde de prediccion
theta_ref = ref2; % MODIFICAR REFERENCIA
for i = 1:numel(Npred_tune)
    t_ref = 0:Ts:(Tf+Npred*Ts); % Vector de tiempo para la referencia a futuro
    ref1 = pi*ones(1,numel(t_ref)); % Referencia 1 (constante = pi)
    ref2 = pi+pi/4*cos(2*pi*freq*t_ref+pi/2); % Referencia 2 (pi + sinusoidal)
    Npred = Npred_tune(i);
    u0 = zeros(Npred, 1);  % Solución propuesta inicial
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
    % Graficar
    plots(t_vec, x_vec, y_vec, u_vec, theta_ref, Ncontrol);
end

%% Parte c) Controlador predictivo fenomenológico
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
m = 1; M = 5; L = 2;

t_vec_interp = 0:0.005:10;
x_vec_interp = interp1(t_vec, x_vec, t_vec_interp, 'spline');% Interpolación con spline para mas fluidez en el gif
plot_gif(t_vec_interp,x_vec_interp,m,M,L,'Control predictivo')

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
m = 1; M = 5; L = 2;

t_vec_interp = 0:0.015:10; % Aumentar la cantidad de frames
% Interpolación con spline para mas fluidez en el gif
x_vec_interp = interp1(t_vec, x_vec, t_vec_interp, 'spline');
plot_gif(t_vec_interp,x_vec_interp,m,M,L,'Control predictivo con incerteza')


