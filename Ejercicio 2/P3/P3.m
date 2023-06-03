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
Npred = 6; % Horizonde de prediccion

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
ref2 = pi+pi/2*cos(2*pi*freq*t_ref+pi/2); % Referencia 2 (pi + sinusoidal)

% Graficamos
figure(1)
plot(t_ref,ref1,'b',t_ref,ref2,'r')
legend('Referencia 1','Referencia 2')
ylim([0,3*pi/2]);
xlabel('Tiempo [s]')
ylabel('Angulo [grados]')
title('Referencias')
grid on
hold off;


%% Parte c) Controlador predictivo fenomenológico
theta_ref = ref1; % MODIFICAR REFERENCIA
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

% Generar gif
m = 1; M = 5; L = 2;
plot_gif(t_vec,x_vec,m,M,L,'Control predictivo')