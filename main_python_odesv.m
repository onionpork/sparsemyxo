function [tB, xB] = main_python_odesv(Xi, polyorder, tspan, x0)

usesine = 0; 
n = 6;
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[tB,xB]=ode45(@(t,x)sparseGalerkin(t,x,Xi,polyorder,usesine),tspan,x0,options);