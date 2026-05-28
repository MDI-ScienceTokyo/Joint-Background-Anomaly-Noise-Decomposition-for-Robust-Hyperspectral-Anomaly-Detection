function X = proxNuclearNorm(X, gamma)

[rows, cols, bands] = size(X);

X = reshape(X, rows*cols, bands)';

[U, S, V] = svd(X,'econ');
S = diag(max(diag(S) - gamma,0));
X = U*S*V';

X = reshape(X', rows, cols, bands);
end

