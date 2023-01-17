function Gama=compute_gama(Alfa, Beta)

    [~, m]=size(Alfa);
    P=diag(Alfa*Beta')*ones(1,m);
    Gama=(Alfa.*Beta) ./P;
end