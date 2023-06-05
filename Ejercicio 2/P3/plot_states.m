%% plot states
function plot_states(t, x, u, ref)
%pos = mod(cumtrapz(t, x(:, 2)),20);
% Crear gráfico con 3 subplots
figure;

% Subplot 1: Estados reales
subplot(5, 1, 1);
plot(t, x(:, 1), 'r-');
xlabel('Tiempo');
ylabel('x');
title('Posición v/s Tiempo');
% Subplot 2: Entrada
subplot(5, 1, 2);
plot(t, x(:, 2), 'r-');
xlabel('Tiempo');
ylabel('v');
title('Velocidad v/s Tiempo');
% Subplot 3: Salida real + Referencia
subplot(5, 1, 3);
legend('Salida', 'Referencia');
xlabel('Tiempo');
ylabel('theta');
title('Ángulo v/s Tiempo');
plot(t, x(:, 3), 'r-', t, ref(1:101), 'b.-');

% Subplot 4: Variación de la entrada
subplot(5, 1, 4);
plot(t, x(:, 4), 'r-');
xlabel('Tiempo');
ylabel('w');
title('Vel. Angular v/s Tiempo');
% Subplot 5: Variación de la entrada
subplot(5, 1, 5);
plot(t(1:100), u, 'r-');
xlabel('Tiempo');
ylabel('Entrada');
title('Entrada v/s Tiempo');
% Ajustar espaciado entre subplots
sgtitle('Resultados del Control Predictivo');
hold off;
end