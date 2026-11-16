function [X, Xclean, tangent, latent, length_gamma] = generate_single_index(N, D, sigma)
%GENERATE_SINGLE_INDEX Segment with independent Gaussian normal coordinates.
if nargin < 3, sigma = 0.5; end
length_gamma = sqrt(12)*sigma*sqrt(D);
latent = sort(rand(1,N));
Xclean = [length_gamma*latent; zeros(D-1,N)];
X = Xclean + [zeros(1,N); sigma*randn(D-1,N)];
tangent = [ones(1,N); zeros(D-1,N)];
end
