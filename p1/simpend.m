clear all, close all; clc
%% Definición de variables relevantes
syms x1 x2 x3 x4 u
m = 1;
M = 5;
L = 2;
g = -9.8;
d = 1;

dt = 0.1;
tspan = 0:dt:10;

%% Simulación del sistema no lineal
y0 = [0; 0; pi-0.1; 0];
[t,y] = ode45(@(t,y)pendcart(y,m,M,L,g,d,0),tspan,y0);%dy = pendcart(y,m,M,L,g,d,u)
plot_gif(t,y,m,M,L,0,-1, 'Sistema No-Lineal Original')

%% Simulación del sistema no lineal (respuesta al escalón)
y0 = [0; 0; pi-0.1; 0];
[t,y] = ode45(@(t,y)pendcart(y,m,M,L,g,d,1),tspan,y0);%dy = pendcart(y,m,M,L,g,d,u)
plot_gif(t,y,m,M,L,heaviside(tspan)', -1, 'Sistema No-Lineal Original - Escalon')

%% Ecuaciones del sistema no lineal
Sx = sin(x3);
Cx = cos(x3);
D = L*(M+m*(1-Cx^2));

dx = [x2;
    (1/D)*(-m*L*g*Cx*Sx + L*(m*L*x4^2*Sx - d*x2) + L*u);
    x4;
    (1/D)*((m+M)*g*Sx - Cx*(m*L*x4^2*Sx - d*x2) - Cx*u)];

%% Linealización del sistema
A = jacobian(dx, [x1, x2, x3, x4]);
B = jacobian(dx, u);

%% Punto de operación y parámetros LQR
% [x, v, theta, dtheta, u]
puntos=[[0, 0, pi, 0, 0];
        [0, 0, pi/2, 0, 0]; % No controlable
        [0, 0, -pi/2, 0, 0]; % No controlable
        [0, 0, 0, 0, 0]];

Q = [1 0  0  0  ;
     0 1  0  0  ;
     0 0 10  0  ;
     0 0  0 100];
 
R = 0.1;

%% Simulación del sistema controlado lineal vs no lineal (π y 0)
K = [];
i = 1;
k_punto = 1;
while i<=size(puntos, 1)
    A_eval = double(subs(A, [x1, x2, x3, x4, u], puntos(i,:)));
    B_eval = double(subs(B, [x1, x2, x3, x4, u], puntos(i,:)));
    % Calcula la matriz de controlabilidad y verifica su rango
    C = ctrb(A_eval, B_eval);
    if rank(C) == size(A_eval, 1)
        Ki = lqr(A_eval,B_eval,Q,R);
        ei = eigs(A_eval-B_eval*Ki);
        display(ei);
        K = [K; Ki];
        disp("El sistema es controlable en el punto de operación " + num2str(k_punto));
        delta = [0 0 0.5 0];
        ref = puntos(i, 1:end-1);
        y0 = ref + delta;
        close all
        % function dy = pendcart(y,m,M,L,g,d,u)
        [t,y] = ode45(@(t,y)pendcart(y,m,M,L,g,d,-Ki*(y - ref')),tspan,y0);
        u_ctr = -Ki*(y' - ref');
        plot_gif(t,y,m,M,L,u_ctr, [ref 0], ['Sistema No-Lineal - Control - Punto de operacion ', num2str(i)])
        %-------------------------------
        [t1,y1] = ode45(@(t1,y1)pendcartlin(A_eval, B_eval, y1, 0),tspan,y0);%dy = pendcart(y,m,M,L,g,d,u)
        plot_states(t1,y1,0,-1, ['Sistema Lineal - Lazo abierto - ','Punto de operacion ', num2str(i)]);
        
        [t2,y2] = ode45(@(t2,y2)pendcartlin(A_eval,B_eval,y2,1),tspan,y0);%dy = pendcart(y,m,M,L,g,d,u)
        plot_states(t2,y2,heaviside(tspan)', -1, ['Sistema Lineal - Escalon - ','Punto de operacion ', num2str(i)]);
        
        [t3,y3] = ode45(@(t3,y3)pendcartlin(A_eval,B_eval,y3,-Ki*(y3 - ref')),tspan,y0);
        u_ctr = -Ki*(y3' - ref');
        plot_states(t3,y3,u_ctr, [ref 0], ['Sistema Lineal - Control - Punto de operacion ', num2str(i)])
        %-------------------------------
        
        pause(5);
        i = i+1;
    else
        disp("El sistema no es controlable en el punto de operación " + num2str(k_punto));
        puntos(i, :) = [];
    end
    k_punto = k_punto+1;
    pause(5);
end

%% Funciones útiles

function plot_gif(t,y,m,M,L,u,ref,title)
    dt = t(2) - t(1);
    filename = strcat(title, '.gif');  % nombre del archivo gif
    figure();
    for k=1:length(t)
        drawpend(y(k,:),m,M,L);
        pause(dt/5);

        % Agregar el cuadro actual a la animación gif
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if k == 2
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
        elseif k > 2
            imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime',dt/5);
        end
    end
    pause(1);
    plot_states(t,y,u, ref, title);
end

function plot_states(t, y, u, ref, title)
    if u ~= 0 % Entrada
        n = 5;
    else
        n = 4;
    end
    %-----------------------------------------------------
    % Graficar los cuatro estados del sistema
    figure;
    hold on;
    for i=1:4
        subplot(n,1,i);
        labels = {'$x$ [m]','$v$ [m/s]', '$\theta$ [rad]', '$\omega$ [rad/s]', '$u$ [N]'};
        plot(t, y(:, i),'b','LineWidth',1);
        if ref ~= -1
            hold on;
            plot(t, ref(i)*ones(size(t,1),1), '--', 'LineWidth', 1, 'Color', 'black');
            name = split(labels(i));
            legend([name(1), 'ref'], 'Location', 'best','Interpreter', 'latex');
        end
        ylabel(labels(i),'Interpreter', 'latex', 'FontSize', 10);
    end
    
    if u ~= 0
        subplot(n,1,n);
        plot(t, u, 'r', 'LineWidth', 1);
        if ref ~= -1
            hold on;
            plot(t, ref(5)*ones(size(t,1),1), '--', 'LineWidth', 1.5, 'Color', 'black');
            name = split(labels(n));
            legend([name(1), 'ref'], 'Location', 'best','Interpreter', 'latex');
        end
        ylabel(labels(5),'Interpreter', 'latex', 'FontSize', 10);
    end
    
    xlabel('Tiempo [s]', 'Interpreter', 'latex', 'FontSize', 10);
    sgtitle(['Respuesta de estados: ', title], 'Interpreter', 'latex', 'FontSize', 10);
    hold off;
end 