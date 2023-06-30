
function y = HVAC_dis(x,u,Ta,w,Ts)
% Parametros
c1=2.508*10^6;
c2=4.636*10^7;
cp=1012;
R=1.7*10^(-3);
Ra=1.3*10^(-3);
delta=0.7;
DeltaT=13;
% Constantes
a1=(cp*(1-delta))/c1;
a2=1/(R*c1);
a3=1/(Ra*c1);
a4=(DeltaT*cp)/c1;
a5=1/c1;
a6=1/(R*c2);
a7=1/c2;
% Calculo del sistema


xdot=zeros(2,1);
xdot(1)=a1*(Ta-x(1)).*u+a2*(x(2)-x(1))+a3*(Ta-x(1))+a4*u+a5.*w(1);
xdot(2)=a6*(x(1)-x(2))+a7*w(2);

% Discretización utilizando la transformada de Euler
x_next = x + xdot * Ts;
    
% Actualización de los estados
x(1) = x_next(1);
x(2) = x_next(2);
    
% Salida del sistema
y = x;


end

