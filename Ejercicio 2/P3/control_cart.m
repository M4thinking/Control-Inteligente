% Importamos la carpeta con las funciones del pendulo
addpath('utils');

t = 0:0.01:10; % Creamos un vector de tiempo de 0 a 10 segundos con paso de 0.01 segundos
% Creamos una referencia oscilante para un angulo entre -10 y 10 grados
ref1 = 10*sin(2*pi*0.05*t);
% Referencia fija de 0 grados
ref2 = 0*ones(1,length(t));

% Creamos una función para resolver control predictivo generalizado
% con un horizonte de 10 pasos y un horizonte de control de 10 pasos

N = 10; % Horizonte de prediccion

% Condiciones iniciales
x0 = [0; 0; 5*pi/4; 0.1]; % estado inicial
u_prev = 0; % valor de u(-1) o u_anterior
z0 = zeros(2*(N+1) + N, 1);
nx = length(x0); % numero de estados
% Variables de control
Ncontrol = 100;
x_control = zeros(nx, Ncontrol+1);
u_control = zeros(1, Ncontrol);
x_control(:,1) = x0;

%%  Loop de control hasta Ncontrol
for k = 1:Ncontrol
    % Funcion objetivo
    fun = @(z) obj_fun(z, xeq, ueq, Q, R, N);
    con = @(z) nonlcon(z, x0, u_prev, A, B, N);
    % Solve optimization problem
    options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
    [z, fval, exitflag, output] = fmincon(fun,z0,[],[],[],[],lb,ub,con,options);
    % Extraer accion de control a aplicar al sistema (primer u)
    u = z(nx*(N+1)+1);
    % Guardar estados y acciones
    u_control(:,k) = u(:,1);
    x_control(:,k+1) = A*x0 + B*u(:,1) + 0.5 * (x0 - xeq).^2;
    % Actualizar estado inicial
    x0 = x_control(:,k+1);
    u_prev = u(:,1);
end