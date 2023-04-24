function aprbs = aprbsGen(Tfinal,Ts)
    % generacion se�al aprbs
    tau = 6;  % especificamos tiempo de establecimiento
    Ntau = tau/Ts; % cuantas muestras dan el t de establecimiento
    % el ancho de cada pulso se mover� entre tau y 1.5 tau
    Amax = 20;  % cotas de la se�al
    Amin = -20;  

    Time = Ts:Ts:Tfinal;
    Nmuestras = length(Time);
    Signal = zeros(1,Nmuestras);

    Nchanges = [1] ; % vector de muestras de cambio
    % partimos en muestra 1
    rng('default')

    i=1;
    while Nchanges(i)+Ntau < Nmuestras
        Nplus = round(Ntau + rand(1)*(Ntau*0.5));
        Nchanges =[Nchanges, Nchanges(end)+Nplus];
        i = i+1;
    end

    % ya tenemos las muestras de cambio
    % ahora procedemos a generar los valores, segun el largo de Nchanges
    Asignal = zeros(1, length(Nchanges)+1);
    for k =1:length(Asignal)
        Asignal(k) = rand(1)*(Amax-Amin)+Amin ;
    end

    % finalmente asignamos a la se�al completa los valores
    for j = 1:length(Nchanges)-1
        for i = Nchanges(j):Nchanges(j+1)-1
            Signal(i) = Asignal(j);
        end
    end
    plot(Time, Signal(1:Nmuestras))
    grid('on')
    title('Señal APRBS de Entrada')
    xlabel('Tiempo')
    ylabel('Entrada')
    aprbs = [Time',Signal(1:Nmuestras)'];
end
