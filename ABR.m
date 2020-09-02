clear all, close all, clc
addpath('./utils');
figpath = '../figures/';

%% generate Data
polyorder = 3;
usesine = 0;
n = 6;  % 3D system
A = [-.1 2 0; -2 -.1 0 ; 0 0 -.3];
dt = 0.01;
tspan=[dt:dt:60];   % time span
x0 = [1, 0, 0, 0, 1, 1];        % initial conditions
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[t, pop] = ode45(@(t,pop)abr_ode(t,pop),tspan,x0,options); 

pop = pop + 0.2 *randn(size(pop));
pop = pop + 0.3 *randn(size(pop));


figure 
plot((pop(2:end-1,1) - pop(3:end,1))/dt)
hold on
dxt(:,1) = TVRegDiff( pop(:,1), 10, .00002, [], 'small', 1e12, dt, 1, 1 );
plot(dxt(:,1))
% for i=1:length(pop)
%     dpop_std(i,:) = abr_ode(t(i), pop(i,:))';
% end
% 
% 
% %% compute Derivative
% 
% dpop = zeros(size(pop));
% dpop(2:end-1,:) = (pop(3:end,:) - pop(1:end-2,:))/ (2*dt);
% dpop(1,:) = (-11/6 * pop(1,:) + 3* pop(2,:) - 3/2*pop(3,:) +pop(4,:)/3)/dt;
% dpop(end,:) = (11/6 * pop(end,:) - 3* pop(end-1,:) + 3/2*pop(end-2,:) - pop(end-3,:)/3)/dt;
% 
% 
% 
% 
% 
% 
% %% pool Data  (i.e., build library of nonlinear time series)
% Theta = poolData(pop,n,polyorder,usesine);
% m = size(Theta,2);
% 
% %% compute Sparse regression: sequential least squares
% lambda = 0.085;      % lambda is our sparsification knob.
% Xi = sparsifyDynamics(Theta,dpop,lambda,n)
% poolDataLIST({'la','lb','lr','ra', 'rb','rr'},Xi,n,polyorder,usesine);
% 
% %% integrate true and identified systems
% [tA,xA]=ode45(@(t,pop)abr_ode(t,pop),tspan,x0,options);   % true model
% [tB,xB]=ode45(@(t,pop)sparseGalerkin(t,pop,Xi,polyorder,usesine),tspan,x0,options);  % approximate
% 
% %% FIGURES!!
% figure
% dtA = [0; diff(tA)];
% plot3(xA(:,1),xA(:,2),xA(:,3),'r','LineWidth',1.5);
% view(49,20)
% hold on
% dtB = [0; diff(tB)];
% plot3(xB(1:5:end,1),xB(1:5:end,2),xB(1:5:end,3),'k--','LineWidth',1.2);
% l1 = legend('True','Identified');
% set(l1,'FontSize',13,'Location','NorthEast');
% set(gca,'FontSize',13);
% xlabel('x_1','FontSize',13)
% ylabel('x_2','FontSize',13)
% zlabel('x_3','FontSize',13)
% view(49,20)
% grid on
% 
% figure
% plot(tA,xA(:,1),'r','LineWidth',1.5)
% hold on
% plot(tA,xA(:,2),'b','LineWidth',1.5)
% plot(tA,xA(:,3),'g','LineWidth',1.5)
% plot(tB,xB(:,1),'k--','LineWidth',1.2)
% plot(tB,xB(:,2),'k--','LineWidth',1.2)
% plot(tB,xB(:,3),'k--','LineWidth',1.2)
% xlabel('Time','FontSize',13)
% ylabel('State, x_k','FontSize',13)
% legend('True x_1','True x_2','True x_3','Identified')
