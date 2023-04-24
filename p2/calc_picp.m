function [ratio] = calc_picp(y, y_inf, y_sup)
    ratio = sum(y_inf<=y & y<=y_sup)/double(length(y));
end