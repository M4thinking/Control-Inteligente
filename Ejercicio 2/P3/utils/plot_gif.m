function plot_gif(t,y,m,M,L,title)
    latency = 1/5;
    dt = t(2) - t(1);
    dt = dt*latency;
    filename = strcat(title, '.gif');  % nombre del archivo gif
    figure();
    for k=1:length(t)
        drawpend(y(k,:),m,M,L);
        pause(dt);
        % Agregar el cuadro actual a la animación gif
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if k == 2
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
        elseif k > 2
            imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime',dt/5);
        end
    end
end