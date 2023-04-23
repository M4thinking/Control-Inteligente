function [yk, zk] = ysim3(z, model)
    % Función que permite encontrar los regresores para y(k|k-1) y z(k|k-1)
    yk = ysim(z, model.a, model.b, model.g);
    regs = size(z,2)/2;
    % zk = y(k|k-1),y(k-1|k-1),u(k-1+1), u(k-2+1).
    % Primera entrada ya no se usa.
    zk = [yk(1:end-1), z(1:end-1, 1), z(2:end, 3), z(2:end, 4)];
    yk = yk(1:end); % Ultima entrada no se puede usar sin saber u actual.