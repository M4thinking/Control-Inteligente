% Importamos la carpeta con las funciones del pendulo
addpath('utils');

% Creamos una referencia oscilante para un angulo entre -10 y 10 grados
ref1 = 10*sin(2*pi*0.05*t);
% Referencia fija de 0 grados
ref2 = 0*ones(1,length(t));

% Creamos una función para resolver control predictivo generalizado
% con un horizonte de 10 pasos y un horizonte de control de 10 pasos

