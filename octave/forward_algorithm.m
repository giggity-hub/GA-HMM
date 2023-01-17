function varargout = forward_algorithm(A,B,O,I)
    [m,n]=size(B);
    N=length(O);

    Alfa=zeros(N,m);
    for k=1:m
        Alfa(1,k)=I(k)*B(k,O(1));
    end

    for l=2:N,
        for k=1:m,
            S=0;
            for i=1:m,
                S=S+A(i,k)*Alfa(l-1,i);
            end
            Alfa(l,k)=B(k,O(l))*S;
        end
    end

    if nargout==1,
        varargout={Alfa};
    else
        varargout(1)={Alfa};varargout (2)={log(P)};
    end
end