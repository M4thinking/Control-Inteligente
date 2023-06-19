clc;clear all
%% a) graficar la función y optimizar mediante fmincon
% Definir los límites de x e y
x = linspace(-10, 10, 100);
y = linspace(-10, 10, 100);

% Crear una malla de puntos x, y
[X, Y] = meshgrid(x, y);

% Calcular los valores de la función en cada punto de la malla
Z = -abs(sin(X).*cos(Y).*exp(abs(1 - (sqrt(X.^2 + Y.^2)/pi))));

% Graficar la función
surf(X, Y, Z);
title('Gráfico de la función');
xlabel('x');
ylabel('y');
zlabel('f(x, y)');

% Definir la función
fun = @(x) -abs(sin(x(1))*cos(x(2))*exp(abs(1 - (sqrt(x(1)^2 + x(2)^2)/pi))));

% Definir los límites de las variables x e y
lb = [-10, -10]; % Límites inferiores
ub = [10, 10];   % Límites superiores

options = optimoptions('fmincon', 'Algorithm', 'sqp'); % Utilizando el método Sequential Quadratic Programming (SQP)

% Ejecutar la optimización
tic; % Iniciar temporizador
[x_opt, fval] = fmincon(fun, [0, 0], [], [], [], [], lb, ub, [], options);
tiempo_ejecucion = toc; % Detener temporizador

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion iniciando en x_0,y_0 = [0, 0]')
disp(['Valor mínimo de f(x, y): ', num2str(fval)]);
disp(['Punto de solución encontrado: [', num2str(x_opt(1)), ', ', num2str(x_opt(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion), ' segundos']);
disp('---------------------------------------------------')

% Ejecutar la optimización segundo punto de inicializacion
tic; % Iniciar temporizador
[x_opt_2, fval_2] = fmincon(fun, [10, -10], [], [], [], [], lb, ub, [], options);
tiempo_ejecucion_2 = toc; % Detener temporizador

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion iniciando en x_0,y_0 = [10, -10]')
disp(['Valor mínimo de f(x, y): ', num2str(fval_2)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_2(1)), ', ', num2str(x_opt_2(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_2), ' segundos']);
disp('---------------------------------------------------')

% Ejecutar la optimización tercero punto de inicializacion
tic; % Iniciar temporizador
[x_opt_3, fval_3] = fmincon(fun, [-5, -5], [], [], [], [], lb, ub, [], options);
tiempo_ejecucion_3 = toc; % Detener temporizador

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion iniciando en x_0,y_0 = [-5, -5]')
disp(['Valor mínimo de f(x, y): ', num2str(fval_3)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_3(1)), ', ', num2str(x_opt_3(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_3), ' segundos']);
disp('---------------------------------------------------')

% Ejecutar la optimización cuarto punto de inicializacion
tic; % Iniciar temporizador
[x_opt_4, fval_4] = fmincon(fun, [-3, 1], [], [], [], [], lb, ub, [], options);
tiempo_ejecucion_4 = toc; % Detener temporizador

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion iniciando en x_0,y_0 = [-3, 1]')
disp(['Valor mínimo de f(x, y): ', num2str(fval_4)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_4(1)), ', ', num2str(x_opt_4(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_4), ' segundos']);
disp('---------------------------------------------------')

% Ejecutar la optimización quinto punto de inicializacion
tic; % Iniciar temporizador
[x_opt_5, fval_5] = fmincon(fun, [0.5, -1], [], [], [], [], lb, ub, [], options);
tiempo_ejecucion_5 = toc; % Detener temporizador

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion iniciando en x_0,y_0 = [0.5, -1]')
disp(['Valor mínimo de f(x, y): ', num2str(fval_5)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_5(1)), ', ', num2str(x_opt_5(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_5), ' segundos']);
disp('---------------------------------------------------')

%% b) Optimizar mediante PSO (aquí pidió el máximo, sus ¬¬)

fun = @(x) abs(sin(x(1))*cos(x(2))*exp(abs(1 - (sqrt(x(1)^2 + x(2)^2)/pi))));

% Definir los límites de las variables x e y
lb = [-10, -10]; % Límites inferiores
ub = [10, 10];   % Límites superiores

% Definir la configuración de PSO
options_pso_1 = optimoptions('particleswarm','MaxIterations', 50, 'SwarmSize',5);

% Ejecutar la optimización con PSO
tic;
[x_opt_pso_1, fval_pso_1] = particleswarm(fun, 2, lb, ub, options_pso_1);
tiempo_ejecucion_pso_1 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con 5 partículas y 50 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_pso_1)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_pso_1(1)), ', ', num2str(x_opt_pso_1(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_pso_1), ' segundos']);
disp('---------------------------------------------------')



% Definir la configuración de PSO
options_pso_2 = optimoptions('particleswarm','MaxIterations', 100, 'SwarmSize',10);

% Ejecutar la optimización con PSO
tic;
[x_opt_pso_2, fval_pso_2] = particleswarm(fun, 2, lb, ub, options_pso_2);
tiempo_ejecucion_pso_2 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con 10 partículas y 100 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_pso_2)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_pso_2(1)), ', ', num2str(x_opt_pso_2(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_pso_2), ' segundos']);
disp('---------------------------------------------------')



% Definir la configuración de PSO
options_pso_3 = optimoptions('particleswarm','MaxIterations', 500, 'SwarmSize',50);

% Ejecutar la optimización con PSO
tic;
[x_opt_pso_3, fval_pso_3] = particleswarm(fun, 2, lb, ub, options_pso_3);
tiempo_ejecucion_pso_3 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con 50 partículas y 500 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_pso_3)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_pso_3(1)), ', ', num2str(x_opt_pso_3(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_pso_3), ' segundos']);
disp('---------------------------------------------------')




% Definir la configuración de PSO
options_pso_4 = optimoptions('particleswarm','MaxIterations', 5000, 'SwarmSize',5);

% Ejecutar la optimización con PSO
tic;
[x_opt_pso_4, fval_pso_4] = particleswarm(fun, 2, lb, ub, options_pso_4);
tiempo_ejecucion_pso_4 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con 5 partículas y 5000 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_pso_4)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_pso_4(1)), ', ', num2str(x_opt_pso_4(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_pso_4), ' segundos']);
disp('---------------------------------------------------')



% Definir la configuración de PSO
options_pso_5 = optimoptions('particleswarm','MaxIterations', 5, 'SwarmSize',5000);

% Ejecutar la optimización con PSO
tic;
[x_opt_pso_5, fval_pso_5] = particleswarm(fun, 2, lb, ub, options_pso_5);
tiempo_ejecucion_pso_5 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con 5000 partículas y 5 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_pso_5)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_pso_5(1)), ', ', num2str(x_opt_pso_5(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_pso_5), ' segundos']);
disp('---------------------------------------------------')


%% c) Optimizar mediante GA

% Definir la función
fun = @(x) -abs(sin(x(1))*cos(x(2))*exp(abs(1 - (sqrt(x(1)^2 + x(2)^2)/pi))));

% Definir los límites de las variables x e y
lb = [-10, -10]; % Límites inferiores
ub = [10, 10];   % Límites superiores

options_ga_1 = optimoptions('ga','MaxGenerations', 50, 'PopulationSize',5);

tic;
[x_opt_ga_1, fval_ga_1] = ga(fun,2,[],[],[],[],lb,ub,[],options_ga_1);
tiempo_ejecucion_ga_1 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con poblacion 5 y 50 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_ga_1)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_ga_1(1)), ', ', num2str(x_opt_ga_1(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_ga_1), ' segundos']);
disp('---------------------------------------------------')



options_ga_2 = optimoptions('ga','MaxGenerations', 100, 'PopulationSize',10);

tic;
[x_opt_ga_2, fval_ga_2] = ga(fun,2,[],[],[],[],lb,ub,[],options_ga_2);
tiempo_ejecucion_ga_2 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con poblacion 10 y 100 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_ga_2)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_ga_2(1)), ', ', num2str(x_opt_ga_2(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_ga_2), ' segundos']);
disp('---------------------------------------------------')



options_ga_3 = optimoptions('ga','MaxGenerations', 500, 'PopulationSize',50);

tic;
[x_opt_ga_3, fval_ga_3] = ga(fun,2,[],[],[],[],lb,ub,[],options_ga_3);
tiempo_ejecucion_ga_3 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con poblacion 50 y 500 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_ga_3)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_ga_3(1)), ', ', num2str(x_opt_ga_3(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_ga_3), ' segundos']);
disp('---------------------------------------------------')



options_ga_4 = optimoptions('ga','MaxGenerations', 5000, 'PopulationSize',5);

tic;
[x_opt_ga_4, fval_ga_4] = ga(fun,2,[],[],[],[],lb,ub,[],options_ga_4);
tiempo_ejecucion_ga_4 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con poblacion 5 y 5000 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_ga_4)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_ga_4(1)), ', ', num2str(x_opt_ga_4(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_ga_4), ' segundos']);
disp('---------------------------------------------------')



options_ga_5 = optimoptions('ga','MaxGenerations', 5, 'PopulationSize',5000);

tic;
[x_opt_ga_5, fval_ga_5] = ga(fun,2,[],[],[],[],lb,ub,[],options_ga_5);
tiempo_ejecucion_ga_5 = toc;

% Mostrar los resultados
disp('---------------------------------------------------')
disp('Optimizacion con poblacion 5000 y 5 iteraciones')
disp(['Valor mínimo de f(x, y): ', num2str(fval_ga_5)]);
disp(['Punto de solución encontrado: [', num2str(x_opt_ga_5(1)), ', ', num2str(x_opt_ga_5(2)), ']']);
disp(['Tiempo de ejecución: ', num2str(tiempo_ejecucion_ga_5), ' segundos']);
disp('---------------------------------------------------')


