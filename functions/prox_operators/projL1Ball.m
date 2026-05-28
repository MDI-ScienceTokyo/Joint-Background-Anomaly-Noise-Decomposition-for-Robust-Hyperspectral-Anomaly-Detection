function[X] = projL1Ball(X,alpha)

x = X(:);
x = max(abs(x)-max(max((cumsum(sort(abs(x),1,'descend'),1)-alpha)./(1:size(x,1))'),0),0).*sign(x);
X = reshape(x,size(X));


