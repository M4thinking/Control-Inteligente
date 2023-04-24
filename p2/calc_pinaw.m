function [PINAW] = calc_pinaw(y, y_inf, y_sup)
    Nd = length(y);
    R = max(y_sup) - min(y_inf);
    PINAW = 1/double(Nd*R) * sum(y_sup-y_inf);
end