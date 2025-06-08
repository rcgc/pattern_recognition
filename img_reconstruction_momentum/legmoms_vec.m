function v = legmoms_vec(F,ord)
%LEGMOMS_VEC vector of Legendre moments of an image
%   v=legmoms_vec(F,ord) computes the vector v of the continuous Legendre
%   moments of the image F, up to order ord.

M = legmoms(F,ord);
idx = fliplr(triu(ones(size(M))));
v = M(logical(idx))';
