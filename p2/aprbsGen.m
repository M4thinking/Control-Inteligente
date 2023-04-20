function aprbs = aprbsGen()
    % generacion señal aprbs
    Ts = 0.1 ; % tiempo de muestreo
    tau = 2;  % especificamos tiempo de establecimiento
    Ntau = tau/Ts; % cuantas muestras dan el t de establecimiento
    % el ancho de cada pulso se moverá entre tau y 1.5 tau
    Amax = 2;  % cotas de la señal
    Amin = -2;  
    Tfinal = 1000; % tiempo total

    Time = 0.1:Ts:Tfinal;
    Nmuestras = length(Time);
    Signal = zeros(1,Nmuestras);

    Nchanges = [1] ; % vector de muestras de cambio
    % partimos en muestra 1
    rng('default')

    i=1;
    while Nchanges(i)+Ntau < Nmuestras
        Nplus = round(Ntau + rand(1)*(Ntau*0.5),0);
        Nchanges =[Nchanges, Nchanges(end)+Nplus];
        i = i+1;
    end

    % ya tenemos las muestras de cambio
    % ahora procedemos a generar los valores, segun el largo de Nchanges
    Asignal = zeros(1, length(Nchanges)+1);
    for k =1:length(Asignal)
        Asignal(k) = rand(1)*(Amax-Amin)+Amin ;
    end

    % finalmente asignamos a la señal completa los valores
    for j = 1:length(Nchanges)-1
        for i = Nchanges(j):Nchanges(j+1)-1
            Signal(i) = Asignal(j);
        end
    end
    plot(Time, Signal(1:Nmuestras))
    grid('on')
    aprbs = [Time',Signal(1:Nmuestras)'];
end
