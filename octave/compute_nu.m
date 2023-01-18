function nu=compute_nu(Gama, B)
    % Return the number of visits to state i
    % m hidden states, n output states and N observations
    % 
    % B - mxn (confusion matrix)
    % nu - mxn matrix

    [~,n]=size(B);
    nu=(sum(Gama)).'*ones(1,n);
end