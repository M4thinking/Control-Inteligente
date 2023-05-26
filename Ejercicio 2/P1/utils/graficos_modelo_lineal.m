%% Simulación del sistema lineal
y0 = [0; 0; pi-0.1; 0];
[t,y] = ode45(@(t,y)pendcartlin(A_eval, B_eval, y, 0),tspan,y0);%dy = pendcart(y,m,M,L,g,d,u)
plot_states(t,y,0,-1, 'Sistema Lineal');
 
%% 
y0 = [0; 0; pi-0.1; 0];
[t,y] = ode45(@(t,y)pendcartlin(A_eval,B_eval,y,1),tspan,y0);%dy = pendcart(y,m,M,L,g,d,u)
plot_states(t,y,heaviside(tspan)', -1, 'Sistema Lineal - Escalon')

%% Simulación del sistema lineal
y0 = [0; 0; 0-0.1; 0];
[t,y] = ode45(@(t,y)pendcartlin(A_eval, B_eval, y, 0),tspan,y0);%dy = pendcart(y,m,M,L,g,d,u)
plot_states(t,y,0,-1, 'Sistema Lineal');
 
%% 
y0 = [0; 0; 0-0.1; 0];
[t,y] = ode45(@(t,y)pendcartlin(A_eval,B_eval,y,1),tspan,y0);%dy = pendcart(y,m,M,L,g,d,u)
plot_states(t,y,heaviside(tspan)', -1, 'Sistema Lineal - Escalon')

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
        K = [K; Ki];
        disp("El sistema es controlable en el punto de operación " + num2str(k_punto));
        delta = [0 0 0.5 0];
        ref = puntos(i, 1:end-1);
        y0 = ref + delta;
        close all
        % function dy = pendcart(y,m,M,L,g,d,u)
        [t,y] = ode45(@(t,y)pendcartlin(A_eval,B_eval,y,-Ki*(y - ref')),tspan,y0);
        u_ctr = -Ki*(y' - ref');
        plot_states(t,y,u_ctr, [ref 0], ['Punto de operacion ', num2str(i)])
        pause(5);
        i = i+1;
    else
        disp("El sistema no es controlable en el punto de operación " + num2str(k_punto));
        puntos(i, :) = [];
    end
    k_punto = k_punto+1;
    pause(5);
end
%% 
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
        labels = {'$x$ [m]','$v$ [m/s]', '$\theta$ [rad]', '$\omega$ [rad/s]', '$u$ [rad/s]'};
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