function dx = pendcart(x,m,M,L,g,d,u)
% x = [x, x_dot, theta, theta_dot]
Sx = sin(x(3));
Cx = cos(x(3));
D = L*(M+m*(1-Cx^2));

dx=[x(2);(1/D)*(-m*L*g*Cx*Sx + L*(m*L*x(4)^2*Sx - d*x(2)) + L*u);...
    x(4);(1/D)*((m+M)*g*Sx - Cx*(m*L*x(4)^2*Sx - d*x(2)) - Cx*u)];
end