function dx = pendcart(x,m,M,L,g,d,u)

Sx = sin(x(3));
Cx = cos(x(3));
D = L*(M+m*(1-Cx^2));

dx(1,1) = x(2);
dx(2,1) = (1/D)*(-m*L*g*Cx*Sx + L*(m*L*x(4)^2*Sx - d*x(2)) + L*u);
dx(3,1) = x(4);
dx(4,1) = (1/D)*((m+M)*g*Sx - Cx*(m*L*x(4)^2*Sx - d*x(2)) - Cx*u);