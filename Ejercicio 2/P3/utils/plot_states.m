%% plot states
function plot_states(t, x, u, ref, Ncontrol)
%pos = mod(cumtrapz(t, x(:, 2)),20);
% Crear gráfico con 3 subplots
figure;

% Subplot 1: Estados reales
subplot(3, 2, 1);
plot(t, x(:, 1), 'r-');
xlabel('Tiempo');
ylabel('x [m]');
title('Posición v/s Tiempo');
% Subplot 2: Entrada
subplot(3, 2, 2);
plot(t, x(:, 2), 'r-');
xlabel('Tiempo');
ylabel('v [m/s]');
title('Velocidad v/s Tiempo');
% Subplot 3: Salida real + Referencia
subplot(3, 2, 3);
plot(t, x(:, 3), 'r-', t, ref(1:Ncontrol+1), 'b.-');
legend('Salida', 'Referencia');
xlabel('Tiempo');
ylabel('theta [rad]');
title('Ángulo v/s Tiempo');
% Subplot 4: Variación de la entrada
subplot(3, 2, 4);
plot(t, x(:, 4), 'r-');
xlabel('Tiempo');
ylabel('w [rad/s]');
title('Vel. Angular v/s Tiempo');
% Subplot 5: Variación de la entrada
subplot(3, 2, 5);
plot(t(1:Ncontrol), u, 'r-');
xlabel('Tiempo');
ylabel('Entrada [N]');
title('Entrada v/s Tiempo');
subplot(3, 2, 6);
plot(t(1:Ncontrol), diff([0;u]), 'r-');
xlabel('Tiempo');
ylabel('Variacion Entrada [N]');
title('Variación Entrada v/s Tiempo');
% Ajustar espaciado entre subplots
sgtitle('Resultados del Control Predictivo');
hold off;
end