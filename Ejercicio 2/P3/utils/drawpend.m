function drawpend(state,m,M,L)
x = state(1);
th = state(3);

% Dimensiones espaciales
W = 1*sqrt(M/5);  % ancho carro
H = .5*sqrt(M/5); % altura carro
wr = .2;          % radio rueda
mr = .3*sqrt(m);  % radio masa

% Posiciones
y = wr/2+H/2; % posicion vertical del carro
pendx = x + L*sin(th);
pendy = y - L*cos(th);

plot([0 40],[0 0],'k','LineWidth',2), hold on % Dibujar linea del piso
rectangle('Position',[x-W/2,y-H/2,W,H],'Curvature',.1,'FaceColor',[.5 0.5 1],'LineWidth',1.5); % Dibujar carro
rectangle('Position',[x-.9*W/2,0,wr,wr],'Curvature',1,'FaceColor',[0 0 0],'LineWidth',1.5); % Dibujar ruerda
rectangle('Position',[x+.9*W/2-wr,0,wr,wr],'Curvature',1,'FaceColor',[0 0 0],'LineWidth',1.5); % Dibujar ruerda
plot([x pendx],[y pendy],'k','LineWidth',2); % Dibujar pendulo
rectangle('Position',[pendx-mr/2,pendy-mr/2,mr,mr],'Curvature',1,'FaceColor',[1 0.1 .1],'LineWidth',1.5); % Dibujar bola

axis([0 40 -2 2.5]);
ylim([-2 2.5]);
axis equal
set(gcf,'Position',[100 100 1000 400])
drawnow, hold off