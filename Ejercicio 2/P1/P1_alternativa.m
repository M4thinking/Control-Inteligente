% Definición de la función objetivo
fun = @(x) -abs(sin(x(1))*cos(x(2))*exp(abs(1-sqrt(x(1)^2+x(2)^2)/pi)));

% Graficación de la función
x1 = linspace(-10, 10, 100);
x2 = linspace(-10, 10, 100);
[X1, X2] = meshgrid(x1, x2);
Z = zeros(length(x1), length(x2));
for i = 1:length(x1)
    for j = 1:length(x2)
        Z(i,j) = fun([x1(i), x2(j)]);
    end
end
surf(X1, X2, Z);
xlabel('x_1');
ylabel('x_2');
zlabel('f(x_1, x_2)');
title('Función objetivo');

% Definición del rango de variables y las restricciones
lb = [-10, -10]; % límite inferior de las variables
ub = [10, 10];   % límite superior de las variables
x0 = [0, 0];     % punto inicial

% Configuración de las opciones del solver
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

% Llamada a fmincon para encontrar el mínimo
tic;
[x_sqp,f_sqp] = fmincon(fun, x0, [], [], [], [], lb, ub, [], options);
execution_time = toc;
% 'SQP terminado en ' + execution_time +' [s]'
disp(['SQP terminado en ', num2str(execution_time), ' [s]']);


% Configuración de las opciones del PSO
options = optimoptions('particleswarm', 'Display', 'iter', 'SwarmSize', 100, 'MaxIterations', 200);

% Llamada a particleswarm para encontrar el máximo
tic;
[x_pso, f_pso] = particleswarm(fun, 2, lb, ub, options); % [x_star, f_star] = particleswarm(fun, 2, lb, ub, options);
execution_time = toc;
disp(['PSO terminado en ', num2str(execution_time),' [s]']);
% Graficación de la trayectoria
figure;
plot(x_star(:,1), x_star(:,2), 'k-o');
xlabel('x_1');
ylabel('x_2');
title('Trayectoria de la solución');

% Graficación de la evolución de la función objetivo
figure;
plot(f_star, 'k-o');
xlabel('Iteración');
ylabel('f(x_1, x_2)');
title('Evolución de la función objetivo');

% Configuración de las opciones del GA
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 100, 'MaxGenerations', 200);

% Llamada a ga para encontrar el máximo
tic;
[x_ga, f_ga] = ga(fun, 2, [], [], [], [], lb, ub, [], options);
execution_time = toc;
disp(['GA terminado en ', num2str(execution_time),' [s]']);

%% Para distintos pares de size y iterations, probar 10 veces cada par y sacar la media (tanto de tiempos como resultados)
experimentos = 10;
sizes = 10:10:100;
iterations = 10:10:100;

resultados_pso = zeros(length(sizes), length(iterations), 4); % 4 -> x1, x2, f, tiempo
resultados_ga = zeros(length(sizes), length(iterations), 4); % 4 -> x1, x2, f, tiempo
for i = 1:length(sizes)
    for j = 1:length(iterations)
        disp(['Size: ', num2str(sizes(i)), ' Iterations: ', num2str(iterations(j))]);
        options_pso = optimoptions('particleswarm', 'Display', 'off', 'SwarmSize', sizes(i), 'MaxIterations', iterations(j));
        options_ga = optimoptions('ga', 'Display', 'off', 'PopulationSize', sizes(i), 'MaxGenerations', iterations(j));
        experimentos_pso = zeros(experimentos, 4);
        experimentos_ga = zeros(experimentos, 4);
        for k = 1:experimentos
            % PSO
            tic;
            [x_pso, f_pso] = particleswarm(fun, 2, lb, ub, options_pso);
            execution_time_pso = toc;
            experimentos_pso(k,1) = x_pso(1);
            experimentos_pso(k,2) = x_pso(2);
            experimentos_pso(k,3) = f_pso;
            experimentos_pso(k,4) = execution_time_pso;
            % GA
            tic;
            [x_ga, f_ga] = ga(fun, 2, [], [], [], [], lb, ub, [], options_ga);
            execution_time_ga = toc;
            experimentos_ga(k,1) = x_ga(1);
            experimentos_ga(k,2) = x_ga(2);
            experimentos_ga(k,3) = f_ga;
            experimentos_ga(k,4) = execution_time_ga;
        end
        resultados_pso(i,j,:) = mean(experimentos_pso, 1);
        resultados_ga(i,j,:) = mean(experimentos_ga, 1);
    end
end

%% Graficación de los resultados (iteraciones vs tamaño, color = f(x1,x2), etiqueta de cada punto = x1,x2, tamaño de cada punto = tiempo)

% PSO
figure;
for i = 1:length(sizes)
    for j = 1:length(iterations)
        scatter(iterations(j), sizes(i), 1000*resultados_pso(i,j,4), resultados_pso(i,j,3), 'filled');
        hold on;
    end
end
xlabel('Iteraciones');
ylabel('Tamaño');
title('PSO');
colorbar( 'southoutside');
hold off;

% GA
figure;
for i = 1:length(sizes)
    for j = 1:length(iterations)
        scatter(iterations(j), sizes(i), 1000*resultados_ga(i,j,4), resultados_ga(i,j,3), 'filled');
        hold on;
    end
end
xlabel('Iteraciones');
ylabel('Tamaño');
title('GA');
colorbar( 'southoutside');
hold off;