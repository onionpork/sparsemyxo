clear all, close all, clc
addpath('./utils');
figpath = '../figures/';

%% generate Data
polyorder = 3;
usesine = 0;
n = 6;  % 3D system
A = [-.1 2 0; -2 -.1 0 ; 0 0 -.3];
rhs = @(x)A*x;   % ODE right hand side
tspan=[0:.1:50];   % time span
x0 = [1, 0, 0, 0, 1, 1];        % initial conditions
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[t, pop] = ode45(@(t,pop)abr_ode(t,pop),tspan,x0,options); 

pop = pop';
%%
X = pop(:,1:end-1);
X2 = pop(:,2:end);
[U,S,V] = svd(X, 'econ');

%%  Compute DMD (Phi are eigenvectors)
r = 3;  % truncate at 21 modes
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);
Atilde = U'*X2*V*inv(S);
[W,eigs] = eig(Atilde);
Phi = X2*V*inv(S)*W;



Anaive = X2* pinv(X);
