function [] = plots(t,x,y,u,ref,nc)
%pos = mod(cumtrapz(t, x(:, 2)),20);
% Crear gráfico con 3 subplots
figure;
% Subplot 1: Estados reales
subplot(4, 1, 1);
plot(t, x(:, 1), 'r-', t, x(:, 2), 'g');
legend('T1', 'T2');
xlabel('Tiempo');
ylabel('Temperaturas');
title('Estados reales');
% Subplot 2: Salida real + Referencia
subplot(4, 1, 2);
plot(t, y, 'b.-', t, ref(1:nc+1), 'r--');
legend('Salida', 'Referencia');
xlabel('Tiempo');
ylabel('Salida');
title('Salida real + Referencia');
% Subplot 3: Entrada
subplot(4, 1, 3);
plot(t(1:end-1), u, 'k');
xlabel('Tiempo');
ylabel('Entrada');
title('Entrada');
% Subplot 4: Variación de la entrada
subplot(4, 1, 4);
plot(t(1:end-2), diff(u), 'k');
xlabel('Tiempo');
ylabel('Variación de la entrada');
title('Variación de la entrada');
% Ajustar espaciado entre subplots
sgtitle('Resultados del Control Predictivo');
hold off;
end