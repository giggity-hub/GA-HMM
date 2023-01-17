function Omega=compute_omega(Gama,B,O)
    [m,n]=size(B);
    for j=1:n,
        inx=find(O==j);
        if ~isempty(inx),
            Omega(:,j)=sum(Gama(inx,:),1).';
        else
            Omega(:,j)=0*ones(m,1);
        end
    end
end