Ntrain = floor(50*0.7);

y=out.salida1(1:Ntrain)';
u=out.entrada1(1:Ntrain)';
Time = 0.1:0.1:3.5;
figure;

% Graficar la primera línea en color rojo
plot(Time, y, 'r');

% Agregar leyenda y etiquetas de los ejes
title('Respuesta al Escalón');
xlabel('Tiempo');
ylabel('Salida');
