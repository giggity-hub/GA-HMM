function taui=compute_taui(Gama,B,O)
    [m,~]=size(B);
    N=length(O);
    taui=Gama(1:N-1,:);
    taui=(sum(taui,1)).'*ones(1,m);
end