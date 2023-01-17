function q=viterbi_algorithm(A,B,O,c)

    [m,n]=size(B);
    N=length(O);

    delta=zeros(N,m);phi=zeros(N,m);
    t=1;
    for k=1:m
        delta(t,k)=c(k)*B(O(t),k);
        phi(t,k)=0
    end

    for t=2:N,
        for k=1:m,
            for l=1:m,
                tmp(l)=delta(t-1,l)*A(l,k)*B(k,O(t));
            end
            [delta(t,k), phi(t,k)] = max(tmp);
        end
    end

    q=zeros(N,1);
    [~,Inx]=max(delta(N,:));
    q(N)=Inx;
    for k=N-1:-1:1,
        q(k)=phi(k+1,q(k+1));
    end
end