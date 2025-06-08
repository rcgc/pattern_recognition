function M = opcmoms(F,P1,P2,usesimpson,fast)
%OPCMOMS numerical computation of the moments of an image
%   M=opcmoms(F,P1,P2) computes the moments of the image F with respect to the
%   basis functions whose values are contained in the columns of P1 and P2.
%
%   M=opcmoms(F,P1,P2,method) chooses between Simpson rule (method=1, default)
%   and trapezoidal rule (method=0).

if nargin<4 || isempty(usesimpson), usesimpson = 1; end
if nargin<5 || isempty(fast), fast = 1; end

[m1 m2] = size(F);
n1 = size(P1,2)-1;
n2 = size(P2,2)-1;

if usesimpson && ~(2*round(m1/2)-m1), error('m1 must be odd.'), end
if usesimpson && ~(2*round(m2/2)-m2), error('m2 must be odd.'), end

if fast
	M = zeros(n1+1,n2+1);
	if usesimpson
		w1 = ones(m1,1); w1(2:2:m1-1)=4; w1(3:2:m1-1)=2;
		h1 = 2/3/(m1-1);
		w2 = ones(m2,1); w2(2:2:m2-1)=4; w2(3:2:m2-1)=2;
		h2 = 2/3/(m2-1);
	else
		w1 = [1 2*ones(1,m1-2) 1]';
		h1 = 1/(m1-1);
		w2 = [1 2*ones(1,m2-2) 1]';
		h2 = 1/(m2-1);
	end
	for k = 1:n1+1
		g = h1*(w1'*(F.*repmat(P1(:,k),1,m2)))';
		M(k,:) = h2*(w2'*(repmat(g,1,n2+1).*P2));
	end
else	% slow
	M = zeros(n1+1,n2+1);
	g = zeros(m1,1);
	for k = 1:n1+1
		for l = 1:n2+1
			if usesimpson
				for i = 1:m2
					g(i) = simpson(F(:,i).*P1(:,k),-1,1);
				end
				M(k,l) = simpson(g.*P2(:,l),-1,1);
			else
				for i = 1:m2
					g(i) = traprule(F(:,i).*P1(:,k),-1,1);
				end
				M(k,l) = traprule(g.*P2(:,l),-1,1);
			end
		end
	end
end

