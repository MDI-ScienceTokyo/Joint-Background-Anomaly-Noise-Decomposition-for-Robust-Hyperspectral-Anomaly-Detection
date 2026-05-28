function result = proxL1Norm(A, gamma)
    result = sign(A).*max(abs(A) - gamma, 0);
end
