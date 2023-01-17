function [A,B,c,LogLik]=baum_welch_multiobs_norm(A,B,O,c)

    [route_nbr, ~] = size(O);
    GamaT=0;
    tauT=0;
    tauiT=0;
    OmegaT=0;
    nuT=0;
    LogLik=0;

    for route=1:route_nbr,
        [Alfa,LP]=forward_algorithm_norm(A,B,O(route,:),c);
        LogLik=LogLik+LP;
        Beta=backward_algorithm_norm(A,B,O(route,:));
        Gama=compute_gama(Alfa,Beta);
        GamaT=GamaT+Gama;
        tau=compute_tau(Alfa,Beta,A,B,O(route,:));
        tauT=tauT+tau;
        taui=compute_taui(Gama,B,O(route(,:)));
        tauiT=tauiT+taui;
        nu=compute_nu(Gama,B);
        nuT=nuT+nu;
        Omega=compute_omega(Gama,B,O(route,:));
        OmegaT=OmegaT+Omega;
    end

    c=GamaT(1,:)/route_nbr;
    A=tauT./tauiT;
    B=OmegaT ./ nuT;
end