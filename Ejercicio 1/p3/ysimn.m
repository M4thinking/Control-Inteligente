function [y, z] = ysimn(z, model, Npred)
    % Entrega y a Npred pasos y z, tal que yk = fTS(z).
    for i = 1:Npred
        if i < Npred
            [y, z] = ysim3(z, model);
        else
            [y, ~] = ysim3(z, model); % Ultimo: No usamos siguiente z
        end
    end
end