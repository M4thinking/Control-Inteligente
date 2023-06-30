%% Seccion a)

% Definición de la función objetivo
fun = @(x) -abs(sin(x(1)) * cos(x(2)) * exp(abs(1 - sqrt(x(1)^2 + x(2)^2)) / pi));

% Restricciones del dominio
lb = [-10, -10];
ub = [10, 10];

% Puntos iniciales 
x0 = [-10 -10; -5 -5; 0 0; 5 5; 10 10];

% Llamada a fmincon para minimizar la función objetivo
options = optimoptions('fmincon', 'Display', 'iter');
tic;
[x_min, fval] = fmincon(fun, x0, [], [], [], [], lb, ub, [], options);
execution_time = toc;

for i=1:length(x0)
 tic;
 [x_min, fval] = fmincon(fun, x0(i,:), [], [], [], [], lb, ub, [], options);
 x_opts(i,:)=x_min;
 f_val(i)=fval;
 execution_time(i) = toc;
end

% Resultados
for i=1:length(x0)
    disp(i)
    fprintf('Condiciones iniciales: %.4f\n', x0(i,:));
    fprintf('Valor de f(x, y): %.4f\n', -f_val(i));
    fprintf('Punto de solución [x*, y*]: [%.4f, %.4f]\n', x_opts(i,:));
    fprintf('Tiempo de ejecución: %.4f segundos\n', execution_time(i));
end

% Graficar la función en el dominio
x = linspace(-10, 10, 100);
y = linspace(-10, 10, 100);
[X, Y] = meshgrid(x, y);
Z = -abs(sin(X) .* cos(Y) .* exp(abs(1 - sqrt(X.^2 + Y.^2)) / pi));
figure;
surf(X, Y, Z);
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
title('Gráfica de f(x, y)');
%% seccion b)

% Configuración del algoritmo PSO
options = optimoptions('particleswarm', 'Display', 'iter', 'SwarmSize', 100, 'MaxIterations', 100);

% Llamada a particleswarm para encontrar el máximo de la función objetivo
tic;
[x_pso, fval_pso] = particleswarm(fun, 2, lb, ub, options);
execution_time_pso = toc;

% Resultados
fprintf('Valor máximo de f(x, y): %.4f\n', -fval_pso);
fprintf('Punto de solución [xPSO*, yPSO*]: [%.4f, %.4f]\n', x_pso(1), x_pso(2));
fprintf('Tiempo de ejecución: %.4f segundos\n', execution_time_pso);


%% parte c)

% Configuración del algoritmo GA
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 100, 'MaxGenerations', 100);

% Llamada a ga para encontrar el máximo de la función objetivo
tic;
[x_ga, fval_ga] = ga(fun, 2, [], [], [], [], lb, ub, [], options);
execution_time = toc;

% Resultados
fprintf('Valor máximo de f(x, y): %.4f\n', -fval_ga);
fprintf('Punto de solución [xGA*, yGA*]: [%.4f, %.4f]\n', x_ga(1), x_ga(2));
fprintf('Tiempo de ejecución: %.4f segundos\n', execution_time);

