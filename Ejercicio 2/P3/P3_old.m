
%% Parte b) Crear dos referencias
t = 0:0.01:100; % Creamos un vector de tiempo de 0 a 10 segundos con paso de 0.01 segundos
% Creamos una referencia oscilante para un angulo entre -10 y 10 grados
ref1 = pi*(1+0.1*sin(pi*0.5*t));
% Referencia fija de 0 grados
ref2 = pi*ones(1,length(t));
% Graficamos
figure(1)
plot(t,ref1,'b',t,ref2,'r')
legend('Referencia 1','Referencia 2')
xlim([0,100]);
ylim([0,4]);
xlabel('Tiempo [s]')
ylabel('Angulo [grados]')
title('Referencias')
grid on

%% Parte c) Controlador predictivo fenomenológico
ruido = 0;
x0 = [0; 0; pi+0.01;0];
out = sim('sistema_controlado.slx'); % Simulación 
usim = out.u;
ysim = out.y;
xsim = out.x;
tsim = out.tout;
%% Gráficos
% Subplots, 1 para los 4 estados y otro para la entrada/salida con 2 ejes (izq y der)
figure(2)
subplot(2,1,1)
plot(tsim,xsim(:,1),'b',tsim,xsim(:,2),'r',tsim,xsim(:,3),'g',tsim,xsim(:,4),'k')
legend('x1','x2','x3','x4')
xlim([0,100]);
xlabel('Tiempo [s]')
ylabel('Estados')
title('Estados')
grid on
subplot(2,1,2)
plot(tsim,usim,'b',tsim,ysim,'r')
legend('u','y')
xlim([0,100]);
xlabel('Tiempo [s]')
ylabel('Entrada/Salida')
title('Entrada/Salida')
grid on

%% Parte f) Uso de incerteza
ruido = 1;