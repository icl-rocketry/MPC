%% Housekeeping %%
clear
clc

%%% Script Description %%%
% Author: Adrian Fok
% Last Updated: 28/2/2024

% This script solves a minimum fuel powered descent trajectory optimisation 
% problem with pin-point landing. This .m script acts a preliminary
% solution to Julia syntax Nihal and I are struggling with

% References: http://www.larsblackmore.com/AcikmeseAAS08.pdf

% Improvements: Initial guess relaxation
%               Use Golden Search Algorithm to find optimal landing time
%               Limited fuel trajectory optimisation problem - minimise
%               landing error

%% Main Script %%

tf = 15; % Reduce time of flight for better trajectories
N = 400;

[trajProb, r, v, m] = setUpProblem1(tf,N);
defaultsolver = solvers(trajProb);

opts = optimoptions(defaultsolver,Display="none",OptimalityTolerance=1e-6);

% Initial Guess - Zeroes
x0.u = zeros(N,3);
x0.S = zeros(N,1);


% Supply better initial guess
% ph = ceil(N/10);
% 
% rng default
% u0 = ones(N,3);
% s0 = ones(N,1);
% 
% u0(3*ph:N-ph, :) = 0;
% u0 = u0 + randn(size(u0));
% s0(3*ph:N-ph) = 0;
% s0 = s0 + randn(size(s0));
% 
% x0.u = u0;
% x0.S = s0;

[sol,fval,eflag,output] = solve(trajProb,x0,Options=opts);


plotter(sol, r, [1500, 0 ,2000],[0, 0 ,0], m, tf, N)


%% Helper Functions %%
function [trajProb, r, v, mass] = setUpProblem1(tf, N)


%%% Time discretisation %%%
dt = tf/N;
t = (0:dt:tf)';


%%% Constants %%%
ge = [-9.81, 0, 0]; % Earth acceleration due to gravity
gm = [-3.74, 0, 0]; % Mars acceleration due to gravity
maxThrust = 3150; % Max thrust of each thruster
numberOfThrusters = 6; % Number of thrusters on lander
cantAngle = 27; % Cant angle of each thrusters in degrees
I_sp = 225; % Specific impulse in 1/s
m_wet = 1905; % Wet mass in kg
%m_dry = 1505; % Dry mass in kg
alpha = 1/(I_sp * abs(ge(1)) * cosd(cantAngle)); % Constant rate of mass depletion
minGlide = deg2rad(4); % Minimum glide angle in degrees, with conversion to radians, 


%%% Problem constraints %%%
rho1 = 0.3 * maxThrust * numberOfThrusters * cosd(cantAngle);
rho2 = 0.8 * maxThrust * numberOfThrusters * cosd(cantAngle); 


%%% Boundary conditions %%%
r0 = [1500, 0 ,2000]; % Initial displacement
v0 = [-75, 0, 100]; % Initial velocity

rf = [0, 0, 0]; % Final displacement
vf = [0, 0, 0]; % Final velocity


%%% Optimisation variables %%%
u = optimvar("u", N, 3); % Thrust 
sigma = optimvar("S", N, 1); % Change of variable

r = optimexpr(N, 3); % Displacement
v = optimexpr(N, 3); % Velocity
z = optimexpr(N, 1); % Mass variable
z0 = optimexpr(N, 1); % Another mass variable
mu1 = optimexpr(N, 1); % Another mass variable
mu2 = optimexpr(N, 1); % Another mass variable


%%% Problem Formualtion %%%
trajProb = optimproblem;
%trajProb.Objective = sum(sigma)*dt;
trajProb.Objective = -z(N);


%%% Translational dynamics model %%%
% Optimisation expressions %
% Direct method used
z0 = fcn2optimexpr(@changez0, z0, m_wet, N, dt, alpha, rho2, 'OutputSize', [N, 1]);
z = fcn2optimexpr(@changez, z, m_wet, N, dt, alpha, sigma,'OutputSize', [N, 1]);
mu1 = fcn2optimexpr(@changeMu, mu1, N, rho1, z0, 'OutputSize', [N, 1]);
mu2 = fcn2optimexpr(@changeMu, mu2, N, rho2, z0, 'OutputSize', [N, 1]); 
v = fcn2optimexpr(@changeV, v, v0, u, gm, N, dt, 'OutputSize', [N, 3]); % Compute velocity at each time step
r = fcn2optimexpr(@changeR, r, r0, u, v, N, dt, 'OutputSize', [N, 3]); % Compute displacement at each time step


% Constraints %
uCon = norm(u(1:N,:))*exp(z(1:N)) >= rho1; % Lower bound for u, decision variable
uCon2 = norm(u(1:N,:))*exp(z(1:N)) <= rho2; % Upper bound for u, decision variable
dispCon = r(1:N,1) >= 0; % The lander is not a tunneler lmao
thrustCon1 = norm(u(1:N,:)) <= sigma(1:N); % This is causing errors, non-linearities in Jacobian
thrustCon2 = mu1(1:N) .* ((1 - (z(1:N) - z0(1:N)) + ((z(1:N) - z0(1:N)).^2) ./ 2)) <= sigma(1:N); % This is causing program to be run a long time
thrustCon3 = mu2(1:N) .* (1 - (z(1:N) - z0(1:N))) >= sigma(1:N); % Sussy constriant
massCon1 = z0(1:N) <= z(1:N);
massCon2 = log(m_wet - alpha*rho1*t(1:N)) >= z(1:N);
socpCon = sqrt(r(1:N,2).^2 + r(1:N,3).^2) <= tan(minGlide) .* r(1:N,1); % Second-Order Cone Constraint


% Boundary Conditions %
initialCon1 = r(1,:) == r0;
initialCon2 = v(1,:) == v0;

endCon1 = r(N,:) == rf;
endCon2 = v(N,:) == vf; 

trajProb.Constraints.uCon = uCon; 
trajProb.Constraints.uCon2 = uCon2;
trajProb.Constraints.dispCon = dispCon;
%trajProb.Constraints.thrustCon1 = thrustCon1;
%trajProb.Constraints.thrustCon2 = thrustCon2;
trajProb.Constraints.thrustCon3 = thrustCon3;
trajProb.Constraints.initialCon1 = initialCon1;
trajProb.Constraints.initialCon2 = initialCon2;
trajProb.Constraints.massCon1 = massCon1;
trajProb.Constraints.massCon2 = massCon2;
trajProb.Constraints.socpCon = socpCon; 
trajProb.Constraints.endCon1 = endCon1;
trajProb.Constraints.endCon2 = endCon2;


% Variable Revert %
mass = exp(z);

end

function plotter(sol,p,p0,pF, m, tf, N)

figure(1)
psol = evaluate(p, sol);
plot3(psol(:,3),psol(:,2),psol(:,1),'rx')
hold on
plot3(p0(3),p0(2),p0(1),'ks')
plot3(pF(3),pF(2),pF(1),'bo')
hold off
view([18 -10])
xlabel("z")
ylabel("y")
zlabel("x")
legend("Steps","Initial Point","Final Point")

figure(2)
msol = evaluate(m, sol);
plot(linspace(0, tf, N), msol)
hold on
hold off
xlabel("Time (s)")
ylabel("Mass (kg)")

figure(3)
asolm = sol.u;
nasolm = sqrt(sum(asolm.^2,2));
plot(linspace(0, tf, N), nasolm, "rx")
xlabel("Time (s)")
ylabel("Norm(acceleration)")

figure(4)
thrust = zeros(N,3);
for ii = 1:N
    thrust(ii,:) = asolm(ii,:) .* msol(ii);
end

plot(linspace(0, tf, N), thrust(:,1))
hold on
plot(linspace(0, tf, N), thrust(:,2))
plot(linspace(0, tf, N), thrust(:,3))
hold off
xlabel("Time (s)")
ylabel("Thrust")
legend("Thrust_x", "Thrust_y", "Thrust_z")



end


%% fcn2optimisation %%

function z0 = changez0(z0, m_wet, N, dt, alpha, rho2)
% Rate of change of z0, a decision variable used in optimisation problem
    z0(1) = log(m_wet);
    for ii = 2:N
        z0(ii) = log(m_wet - alpha*rho2*dt*(ii-1));
    end
end

function mu = changeMu(mu, N, rho, z0)
    for ii = 1:N
        mu(ii) = rho * exp(-z0(ii));
    end
end

function z = changez(z, m_wet, N, dt, alpha, sigma)
% Rate of change of z0, a decision variable used in optimisation problem
    z(1) = log(m_wet);
    for ii = 2:N
        z(ii) = z(ii-1) - alpha * dt/2 * (sigma(ii-1) + sigma(ii));
    end
end

function v = changeV(v, v0, u, g, N, dt)
% Discretised change in velocity
    v(1,:) = v0;
    for ii = 2:N
        v(ii, :) = v(ii-1, :) + ((u(ii-1, :) + u(ii, :))./2 + g) .* dt;
    end
end

function r = changeR(r, r0, u, v, N, dt)    
% Discretised change in velocity
    r(1,:) = r0;
    for ii = 2:N
        r(ii, :) = r(ii-1, :) + (v(ii,:) + v(ii-1,:)) .* (dt/2) + (u(ii,:)+u(ii-1,:)).*(dt^2/12); 
    end
end