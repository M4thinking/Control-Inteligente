function dx = pendcart(u,x)
% Parámetros del sistema
m = 1;
M = 5;
L = 2;
g = -9.8;
d = 1;

% Simplificación
Sx = sin(x(3));
Cx = cos(x(3));
D = L*(M+m*(1-Cx^2));

% Derivadas del estado. x=[x,dx,theta,dtheta]
dx=[x(2);
    (-m*L*g*Cx*Sx + L*(m*L*x(4)^2*Sx - d*x(2)) + L*u)/D;...
    x(4);
    ((m+M)*g*Sx - Cx*(m*L*x(4)^2*Sx - d*x(2)) - Cx*u)/D
    ];
dx = dx + x;
end