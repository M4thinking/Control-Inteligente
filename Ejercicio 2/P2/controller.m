function u = controller(r, T1, T2, Ta_prev, a,b,g)
uprox0 = ones(1,5); % uprox0 es la condición inicial del optimizador
% Se miniza respecto a u. fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon)
lb = 0.1*ones(1,5);
ub = 2*ones(1,5);
uprox = fmincon(@(uprox)objective(uprox, a,b,g,[T1;T2],r,Ta_prev,1), uprox0, [], [], [], [], lb, ub, []);
% uprox = ga(@(uprox)objective(uprox, a,b,g,[T1;T2],r,Ta_prev,1), 5, [], [], [], [], lb, ub, [], [1,2,3,4,5]);
% uprox = particleswarm(@(uprox)objective(uprox, a,b,g,[T1;T2],r,Ta_prev,1), 5, lb, ub);
% uprox = [u(k+9), u(k+8), ..., u(k)]
u = uprox(1);