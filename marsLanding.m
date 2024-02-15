%% Housekeeping %%
clear
clc

tf = 72;
N = 2000;

[trajProb, r, m] = setUpProblem1(tf,N);
defaultsolver = solvers(trajProb);

opts = optimoptions(defaultsolver,Display="none",OptimalityTolerance=1e-9);
x0.T = zeros(N-1,3);
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

% r = optimexpr(N, 3); % displacement
v = optimexpr(N, 3); % Velocity

Tnorm = sqrt(sum(thrust(1:N-1,:).^2,2));

%%% Translational dynamics model %%%
m = m_wet -  dt * alpha * cumsum(Tnorm);
% Cost for acceleraiton changes

%%% Problem Formualtion %%%
trajProb = optimproblem;
trajProb.Objective = sum(Tnorm)*dt;


% Constraints %
mbig = repmat(m,1,3);
gbig = repmat(gm,size(thrust,1),1);
thrustCon1 = rho1 <= norm(thrust);
thrustCon2 = rho2 >= norm(thrust);  

v = cumsum([v0; dt*(thrust./mbig + gbig)]); % Compute velocity at each time step

socpCon = secondordercone(S, 0, -c', 0); % SOCP Constraint, Error here

r = cumsum([r0; dt*(v(2:N,:) + v(1:(N-1),:))/2]); % Compute displacement at each time step

dispCon = 0 <= r(:,1);

initialCon1 = r(1,:) == r0;
initialCon2 = v(1,:) == v0;

endCon1 = v(N,:) == vf; % Final velocity vector
endCon2 = r(N,:) == rf; % Final displacement vector

trajProb.Constraints.dispCon = dispCon;
trajProb.Constraints.socpCon = socpCon;
trajProb.Constraints.thrustCon = thrustCon1;
trajProb.Constraints.thrustCon = thrustCon2;
trajProb.Constraints.initialCon1 = initialCon1;
trajProb.Constraints.initialCon2 = initialCon2;
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