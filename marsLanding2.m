%% Housekeeping %%
clear
clc

tf = 72;
N = 100 ;

[trajProb, r, m] = setUpProblem1(tf,N);
defaultsolver = solvers(trajProb);

opts = optimoptions(defaultsolver,Display="none",OptimalityTolerance=1e-9);
x0.T = zeros(N-1,3);
x0.S = zeros(N-1,1);
[sol,fval,eflag,output] = solve(trajProb,x0,Options=opts);

plottrajandaccel(sol,r,[1500, 0 ,2000],[0, 0 ,0])
plotMass(sol, m, tf, N-1)


%% Helper Functions %%
function [trajProb, r, m] = setUpProblem1(tf, N)

%%% Time discretisation %%%
dt = tf/N;

%%% Constants %%%
ge = [-9.81, 0, 0];
gm = [-3.74, 0, 0];
maxThrust = 3100;
numberOfThrusters = 6;
cantAngle = 27;
I_sp = 225;
m_wet = 1905;
m_dry = 1505;
alpha = 1/(I_sp * abs(ge(1)) * cosd(cantAngle));
minGlide = deg2rad(86);


%%% Problem constraints %%%
rho1 = 0.3 * maxThrust * numberOfThrusters;
rho2 = 0.8 * maxThrust * numberOfThrusters; 

S = [0, 1, 0, 0, 0, 0;
    0, 0, 1, 0, 0 ,0];
c = [-tan(minGlide), 0 ,0, 0, 0, 0]';


%%% Boundary conditions %%%
r0 = [1500, 0 ,2000]; % Initial displacement
v0 = [-75, 0, 100]; % Initial velocity

rf = [0, 0, 0];
vf = [0, 0, 0];


%%% Optimisation variables %%%
thrust = optimvar("T", N-1, 3, "LowerBound", rho1, "UpperBound", rho2); % Thrust 
sigma = optimvar("S",N-1, 1);

% r = optimexpr(N, 3); % displacement
v = optimexpr(N, 3); % Velocity
gamma = optimexpr(N, 1); % Slack variable
u = optimexpr(N, 3); % Change of variable


%%% Translational dynamics model %%%
dz = -alpha * cumsum(sigma);
z = log(m_wet) + cumsum(dz) * dt;
z0 = log(m_wet - alpha*rho2*dt);
m = exp(z);

gamma = sqrt(sum(thrust(1:N-1,:).^2,2));
mbig = repmat(m,1,3);
u = thrust./mbig;


%%% Problem Formualtion %%%
trajProb = optimproblem;
trajProb.Objective = sum(sigma)*dt;


% Constraints %

gbig = repmat(gm,size(thrust,1),1);
thrustCon1 = sqrt(sum(u(1:N-1,:).^2,2)) <= sigma(1:N-1);
thrustCon2 = rho1*exp(-z0)*(1-(z-z0)+((z-z0).^2)/2) <= sigma;
thrustCon3 = rho2*exp(-z0)*(1-(z-z0)) >= sigma;
massCon1 = z0 <= z;
massCon2 = log(m_wet - alpha*rho1*dt) >= z;
    
v = cumsum([v0; dt*(thrust./mbig + gbig)]); % Compute velocity at each time step
r = cumsum([r0; dt*(v(2:N,:) + v(1:(N-1),:))/2]); % Compute displacement at each time step

dispCon = 0 <= r(:,1);

initialCon1 = r(1,:) == r0;
initialCon2 = v(1,:) == v0;

endCon1 = v(N,:) == vf; % Final velocity vector
endCon2 = r(N,:) == rf; % Final displacement vector

trajProb.Constraints.dispCon = dispCon;
trajProb.Constraints.thrustCon1 = thrustCon1;
trajProb.Constraints.thrustCon2 = thrustCon2;
trajProb.Constraints.thrustCon3 = thrustCon3;
trajProb.Constraints.initialCon1 = initialCon1;
trajProb.Constraints.initialCon2 = initialCon2;
trajProb.Constraints.massCon1 = massCon1;
trajProb.Constraints.massCon2 = massCon2;
trajProb.Constraints.endCon1 = endCon1;
trajProb.Constraints.endCon2 = endCon2;

end

function plottrajandaccel(sol,p,p0,pF)
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
end

function plotMass(sol,m, tf, N)
figure(2)
psol = evaluate(m, sol);
plot(linspace(0, tf, N), psol)
hold on
hold off
xlabel("Time (s)")
ylabel("Mass (kg)")
end