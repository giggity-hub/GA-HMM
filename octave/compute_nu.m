function nu=compute_nu(Gama, B)
    [~,n]=size(B);
    nu=(sum(Gama)).'*ones(1,n)
end