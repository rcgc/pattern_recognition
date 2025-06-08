function [alfa,beta] = opcoef(type,n,N)
%OPCOEF	recursion coefficients for orthogonal polynomials
%   [alpha,beta]=opcoef(type,n) returns the three-term recursion coefficients
%   for the first n+1 orthogonal polynomials of some families, as specified by
%   type: 'Legendre', 'DChebyshev' (discrete Chebyshev), 
%         'Cheb2' (Chebyshev second kind)
%
%   If type='DChebyshev', a third parameter N, specifying the number of points,
%   is required. This family is orthogonal in [0,N-1], the other ones in
%   [-1,1].

if ischar(type)
	switch lower(type(1:5))
	case 'legen'	% Legendre
		type = 1;
	case 'dcheb'	% discrete Chebyshev
		type = 2;
	case 'cheb2'	% Chebyshev second kind
		type = 3;
	otherwise
		error('unknown type.')
	end
end

if type == 2
	if nargin<3, error('the number of points is needed.'), end
	if n+1>N, error('n must be lower than N-1.'), end
end

vk = [1:n+1]';

switch type

case 1		% Legendre
	beta0 = 2;
	alfa = zeros(n+1,1);
	beta = [beta0; vk.^2./(4*vk.^2-1)];

case 2		% discrete Chebyshev
	beta0 = N;
	alfa = (N-1)/2*ones(n+1,1);
	beta = [beta0; N^2/4*(1-(vk./N).^2)./(4-1./(vk.^2))];

case 3		% Chebyshev second kind
	beta0 = pi/2;
	alfa = zeros(n+1,1);
	beta = [beta0; ones(n+1,1)/4];

otherwise
	error('unknown type.')
end

