function [A,B,c]=baum_welch(A,B,O,c)
    Alfa=forward_algorithm(A,B,O,c)
    Beta=backward_algorithm(A, B, O)
    Gama=compute_gama(Alfa,Beta);
    tau=compute_tau(Alfa,Beta,A,B,O);
    taui=compute_taui(Gama,B, O);
    nu=compute_nu(Gama,B);
    Omega=compute_omega(Gama,B,O);
    c=Gama(1,:);
    A=tau ./ taui;
    B=Omega ./ nu;
end