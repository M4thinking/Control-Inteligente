% clear
clc

addpath("Toolbox TS NN/Toolbox NN")
%% Generación APRBS
aprbs = aprbsGen();
%% Correr simulink
out = sim('ident_model.slx');
%% Cargar datos
% load('datos_ejemplo.mat')
% 
% % Se eliminan datos NaN, necesario solo para estos datos
% nanx = isnan(datos);
% t    = 1:numel(datos);
% datos(nanx) = interp1(t(~nanx), datos(~nanx), t(nanx));
% 
% figure()
% plot(datos)
% title('Temperatura ambiental')
% xlabel('Muestras')
% ylabel('Grados [°C]')
%% Parametros modelo
max_regs = 36;
max_regs_list = 1:max_regs;
max_hlayer = 5:5:30;
porcentajes = [20 20 60];

% Se cargan el vector Y de salida y la matriz X de regresores
% Recordar que el orden de Y,X fue elegido arbitrariamente y su forma
% dependera de cada implementacion

% [y, x] = autoregresores(datos, max_regs);
% 
% [Y.val , Y.test, Y.ent, X.val, X.test, X.ent] = separar_datos(y, x, porcentajes);
% load("autorregresores.mat")

porcentajes=[0.6,0.2,0.2];
[y ,x]=autoregresores(out.entrada,out.salida,max_regs);
[Y_val , Y_test, Y_ent, X_val, X_test, X_ent] = separar_datos(y, x, porcentajes);


%% Optimizar modelo - Reglas

% Se calcula el error de test para todas las configuraciones de neuronas en
% capa oculta
% Aqui se calcula el error solo con 15, que fue el optimo precalculado
% Se debe encontrar el optimo mediante iteraciones

net_ent = fitnet(15); % 15 neuronas en capa oculta
net_ent.trainFcn = 'trainscg'; % Funcion de entrenamiento
net_ent.trainParam.showWindow=1; % Evita que se abra la ventana de entrenamiento
net_ent = train(net_ent,X_ent',Y_ent', 'useParallel','yes');

y_p_test = net_ent(X_test')'; % Se genera una prediccion en conjunto de test
errtest= (sqrt(sum((y_p_test-Y_test).^2)))/length(Y_test); % Se guarda el error de test

optim_hlayer = 15;
%% Optimizar modelo - Regresores
[p, indices] = sensibilidad_nn(X_ent, net_ent);

x_optim_ent = X_ent;
x_optim_val = X_val;

% Se quita el regresor con menor sensibilidad
x_optim_ent(:, p) = [];
x_optim_val(:, p) = [];

%% Entrenar modelo
net_optim = fitnet(optim_hlayer);
net_optim.trainFcn = 'trainscg';  
net_optim.trainParam.showWindow=0;
net_optim = train(net_optim,x_optim_ent',Y_ent', 'useParallel','yes');

%% Predicciones
net_optim_structure = my_ann_exporter(net_optim);
y_hat = my_ann_evaluation(net_optim_structure, x_optim_val');
% y_hat = net_optim(x_optim_val')';

figure()
plot(Y_val, '.b')
hold on
plot(y_hat, 'r')

legend('Valor real', 'Valor esperado')

