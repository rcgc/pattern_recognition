function v = dchebmoms_vec(F,ord)
%DCHEBMOMSS_VEC vector of discrete Chebyshev moments of an image
%   v=dchebmoms_vec(F,ord) computes the vector v of the discrete Chebyshev
%   moments of the image F, up to order ord.

M = dchebmoms(F,ord);
idx = fliplr(triu(ones(size(M))));
v = M(logical(idx))';
