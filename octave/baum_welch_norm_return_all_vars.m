function [A,B,c,LogLik, Alfa, Beta, Gama, tau, taui, nu, Omega]=baum_welch_norm(A,B,O,c)
    % A - mxm (state transitions matrix)
    % B = nxm (confusions matrix)
    % O = 1xN (observations vector)
    % c = 1xm (priors vector) 
    [Alfa,LogLik]=forward_algorithm_norm(A,B,O,c);
    Beta=backward_algorithm_norm(A,B,O);
    Gama=compute_gama(Alfa,Beta);
    tau=compute_tau(Alfa,Beta,A,B,O);
    taui=compute_taui(Gama,B,O);
    nu=compute_nu(Gama,B);
    Omega=compute_omega(Gama,B,O);

    c=Gama(1,:);
    A=tau ./ taui;
    B=Omega ./ nu;
end