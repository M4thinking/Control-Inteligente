function sens = sensibilidad_regresores(net, X, Y)
    % Sensibilidad = media(dEdx)^2 + var(dEdx)
    dydx = zeros(size(X,2), size(X,1), size(Y,1));
    for i = 1:size(X,2)
        dth = (1 - tanh(net.IW{1}*X(:,i) + net.b{1}).^2);
        %  La dervidada es la dy/dx = dth * LW{2,1}
        p = net.LW{2,1}'.* dth;
        dydx(i, :, :) = net.IW{1}'*p;
    end
    % Calculo la sensibilidad de cada regresor para cada salida
    sens = zeros(size(X,1), size(Y,1));
    for i = 1:size(X,1)
        for j = 1:size(Y,1)
            sens(i,j) = mean(dydx(:,i,j))^2 + std(dydx(:,i,j))^2;
        end
    end
end