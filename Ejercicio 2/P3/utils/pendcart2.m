function dx = pendcart(x,u)
    % x = [x, x_dot, theta, theta_dot]
    m = 1;
    M = 5;
    L = 2;
    g = -9.8;
    d = 1;
    dx = pendcart(x,m,M,L,g,d,u);