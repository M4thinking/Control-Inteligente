Ntrain = floor(1000);

y=out.salida(1:Ntrain)';
u=out.entrada(1:Ntrain)';
Time = 0.1:0.1:100;
figure;

% Graficar la primera línea en color rojo
plot(Time, y, 'r');

% Agregar leyenda y etiquetas de los ejes
title('Respuesta a señal APRBS');
xlabel('Tiempo');
ylabel('Salida');