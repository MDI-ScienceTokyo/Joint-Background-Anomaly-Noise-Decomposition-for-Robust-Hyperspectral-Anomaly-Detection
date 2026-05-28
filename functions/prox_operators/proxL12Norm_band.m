function result = proxL12Norm_band(X, gamma)
    
    T = max(1 - gamma./sqrt(sum(sum(X.*X, 4), 3)), 0);
    result = T.*X;

end