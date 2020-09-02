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
% options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[t, pop] = ode45(@(t,pop)ABR(t,pop),tspan,x0); 




[x1_val, x1_idx] = datasample(pop(:,1), floor(length(pop(:,1))/20));
[x2_val, x2_idx] = datasample(pop(:,2), floor(length(pop(:,1))/20));
[x3_val, x3_idx] = datasample(pop(:,3), floor(length(pop(:,1))/20));
[x4_val, x4_idx] = datasample(pop(:,4), floor(length(pop(:,1))/20));
[x5_val, x5_idx] = datasample(pop(:,5), floor(length(pop(:,1))/20));
[x6_val, x6_idx] = datasample(pop(:,6), floor(length(pop(:,1))/20));




t_species_matrix = nan(length(t), 6);

t_species_matrix(x1_idx,1) = x1_val;
t_species_matrix(x2_idx,2) = x2_val;
t_species_matrix(x3_idx,3) = x3_val;
t_species_matrix(x4_idx,4) = x4_val;
t_species_matrix(x5_idx,5) = x5_val;
t_species_matrix(x6_idx,3) = x6_val;

t_species_est = funkSVD(t_species_matrix, 10, 0.005, 1000);


x = t_species_est;



% %% compute Derivative
% eps = .01;
% for i=1:length(x)
%     dx(i,:) = A*x(i,:)';
% end
% dx = dx + eps*randn(size(dx));

dx = diff(x, 2)/ .01; 
x = x(2:end-1,:);

%% pool Data  (i.e., build library of nonlinear time series)
Theta = poolData(x,n,polyorder,usesine);
m = size(Theta,2);

%% compute Sparse regression: sequential least squares
lambda = 0.085;      % lambda is our sparsification knob.
Xi = sparsifyDynamics(Theta,dx,lambda,n)
poolDataLIST({'la','lb','lr', 'ra', 'rb', 'rr'},Xi,n,polyorder,usesine);

%% integrate true and identified systems
[tA,xA]=ode45(@(t,pop)ABR(t, pop),tspan,x0);   % true model
[tB,xB]=ode45(@(t,x)sparseGalerkin(t,x,Xi,polyorder,usesine),tspan,x0);  % approximate

%% FIGURES!!
figure
dtA = [0; diff(tA)];
plot3(xA(:,1),xA(:,2),xA(:,3),'r','LineWidth',1.5);
view(49,20)
hold on
dtB = [0; diff(tB)];
plot3(xB(1:5:end,1),xB(1:5:end,2),xB(1:5:end,3),'k--','LineWidth',1.2);
l1 = legend('True','Identified');
set(l1,'FontSize',13,'Location','NorthEast');
set(gca,'FontSize',13);
xlabel('x_1','FontSize',13)
ylabel('x_2','FontSize',13)
zlabel('x_3','FontSize',13)
view(49,20)
grid on

figure
plot(tA,xA(:,1),'r','LineWidth',1.5)
hold on
plot(tA,xA(:,2),'b','LineWidth',1.5)
plot(tA,xA(:,3),'g','LineWidth',1.5)
plot(tB,xB(:,1),'k--','LineWidth',1.2)
plot(tB,xB(:,2),'k--','LineWidth',1.2)
plot(tB,xB(:,3),'k--','LineWidth',1.2)
xlabel('Time','FontSize',13)
ylabel('State, x_k','FontSize',13)
legend('True x_1','True x_2','True x_3','Identified')


function dpop = ABR(t, pop)
ka = 400; da =2; dab = 400;
kb = 2; kbb = 30;  dba = 30;
db = 2.8; K = 0.3;
kr = 0.1; krb = 1.5; dr = 0.2;

la = pop(1);
lb = pop(2);
lr = pop(3);
ra = pop(4);
rb = pop(5);
rr = pop(6);

dpop(1,1) = ka*(1-la-ra)*lr - da*la - dab*la*lb^2;
dpop(2,1) = (1-lb-rb)*(kb + kbb*lb) - db*lb - dba*la*lb^2;
dpop(3,1) = (1-lr-rr)*(kr + krb*lb) - dr*lr;
dpop(4,1) = ka*(1-la-ra)*rr - da*ra - dab*ra*rb^2;
dpop(5,1) = (1-lb-rb)*(kb + kbb*rb) - db*rb - dba*ra*rb^2;
dpop(6,1) = (1-lr-rr)*(kr + krb*rb) - dr*rr;
end