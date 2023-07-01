function u = controller(r, T1, T2, Ta_prev, a,b,g)
uprox0 = ones(1,5); % uprox0 es la condición inicial del optimizador
% Se miniza respecto a u. fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon)
lb = 0.1*ones(1,5);
ub = 2*ones(1,5);
% Fmincon
% medir tiempo
tic
uprox = fmincon(@(uprox)objective(uprox, a,b,g,[T1;T2],r,Ta_prev,1), uprox0, [], [], [], [], lb, ub, []);
% Algoritmo Genético
%options_ga = optimoptions('ga','MaxGenerations', 500, 'PopulationSize',50);
%uprox = ga(@(uprox)objective(uprox, a,b,g,[T1;T2],r,Ta_prev,1), 5, [], [], [], [], lb, ub, [], options_ga);
% Particle Swarm Optimization
%options_pso = optimoptions('particleswarm','MaxIterations', 500, 'SwarmSize',50);
%uprox = particleswarm(@(uprox)objective(uprox, a,b,g,[T1;T2],r,Ta_prev,1),5, lb, ub, options_pso);
toc
u = uprox(1);
end