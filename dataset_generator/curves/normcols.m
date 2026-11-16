function [Xnorm, norm_cols] = normcols( X,p )

if nargin<2 || isempty(p), p = 2; end

% Normalizes the columns of X in the p-norm
norm_cols = (sum(abs(X).^p,1)).^(1/p);

Xnorm = bsxfun(@rdivide, X, norm_cols);

Xnorm(:,norm_cols<eps) = 0;

return
