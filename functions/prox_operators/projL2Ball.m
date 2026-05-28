function result = projL2Ball(X, V, epsilon)
    l2distance = sqrt(sum((X - V).^2, "all"));
    if l2distance <= epsilon
        result = X;
    else
        result = V + epsilon*(X - V)/l2distance;
    end
end
