function [yk, zk] = ysim3(z, model)
    % Función que permite encontrar los regresores para y(k|k-1) y z(k|k-1)
    yk = ysim(z, model.a, model.b, model.g);
    regs = size(z,2);
    % zk = y(k|k-1),y(k-1|k-1),u(k-1+1), u(k-2+1).
    % y se corre a la derecha y se le quita la ultima entrada.
    % u se eliminan primeras entradas.
    zk = [yk(1:end-1), z(1:end-1, 1:regs-1)];
    yk = yk(1:end); % Ultima entrada no se puede usar sin saber u actual.