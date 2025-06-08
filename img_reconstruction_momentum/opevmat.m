function P = opevmat(alfa,beta,x,ortho)
%OPEVMAT evaluation of orthogonal polynomials
%   P=opevmat(alpha,beta,x) computes the values of the orthogonal polynomials
%   defined by the recursion coefficients alpha and beta on the vector of
%   points x. The j-th column of P contains the values of the polynomial of
%   degree j-1.
%
%   P=opevmat(alpha,beta,x,ortho) specifies the normalizazion of the
%   polynomials: ortho=1 (default) evaluates orthonormal polynomials, ortho=0
%   is for monic polynomials.

if nargin<4, ortho = 1; end

n = length(alfa)-1;
m = length(x);
x = x(:);
P = ones(m,n+1);

if ortho	% orthonormal
	P(:,1) = P(:,1) / sqrt(beta(1));
	P(:,2) = (x-alfa(1)).*P(:,1);
	P(:,2) = P(:,2) / sqrt(beta(2));
	for k = 2:n
		P(:,k+1) = (x-alfa(k)).*P(:,k) - sqrt(beta(k)).*P(:,k-1);
		P(:,k+1) = P(:,k+1) / sqrt(beta(k+1));
	end
else		% monic
	P(:,2) = x-alfa(1);
	for k = 2:n
		P(:,k+1) = (x-alfa(k)).*P(:,k) - beta(k).*P(:,k-1);
	end
end

