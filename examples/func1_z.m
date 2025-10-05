function z_output = func1_z(x, y)
    z_output = abs(x) + abs(y.^2) + abs(x.*y);
end