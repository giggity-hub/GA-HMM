function tau=compute_tau(Alfa,Beta,A,B,O)
    [m,n]=size(B);
    N=length(O);
    tau=zeros(m,m);
    for k=1:N-1,
        num=A.*(Alfa(k,:).'*Beta(k+1,:)).*(B(:,O(k+1))*ones(1,m)).';
        den=ones(1,m)*num*ones(m,1);
        tau=tau+num/den;
    end
end