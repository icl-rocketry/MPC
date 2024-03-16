using JuMP
using LinearAlgebra
using Ipopt
using Plots
using Interpolations
using InfiniteOpt

# Model degrees of freedom definition
Dim = 1:3 # Number of dimensions

# Time discretisation
t0 = 0   # Intial time
tf = 72  # Final time
N = 1000 # Number of points (mesh size)

# Planet parameters
ge = [-9.807,0,0]  # Earth's gravity (m/s²)
gm = [-3.7114,0,0] # Mars' gravity (m/s²)

# Spacecraft propulsion parameters
maxThrust = 3100              # Maximum thrust (N)
nThrusters = 6                # Number of thrusters
ϕ = deg2rad(27)               # Cant angle of thrusters
Isp = 225                     # Specific Impulse (s)
α = 1/(Isp*abs(gm[1])*cos(ϕ)) # Constant for mass loss rate

# Mass conditions
m_wet = 1905 # Wet mass of spacecraft (kg)
m_dry = 1505 # Dry mass of spacecraft (kg)

# SOCP parameter
minGlide = deg2rad(86) # Minimum glide angle / Cone angle
S = [0 1 0 0 0 0; 0 0 1 0 0 0]
c = [-tan(minGlide); 0; 0; 0; 0; 0]

# Position conditions numerical values
r_init = [1500, 0, 2000]  # Initial position (m)
r_final = [0, 0, 0]       # Final position (m)
rdot_init = [-75, 0, 100] # Initial velocity (m/s)
rdot_final = [0, 0, 0]    # Final velocity (m/s)

# Thurst constraints numerical values
ρ₁ = 0.3 * maxThrust * nThrusters # Lower thrust constraint
ρ₂ = 0.8 * maxThrust * nThrusters # Upper thrust constraint  

# Create JuMP Model - Infinite Time Horizon Optimization Solver
model = InfiniteModel(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

# Define infinte parameters
@infinite_parameter(model, t in [t0, tf], num_supports = N)

@variables(model, begin
    # state variables
    z, Infinite(t)
    r[Dim], Infinite(t)
    rdot[Dim], Infinite(t)

    # control variables
    σ, Infinite(t)
    T[Dim], Infinite(t), (start = t0)
end)

@objective(model, Min, integral(σ, t))

# Inital Conditions
@constraint(model, c1, z(t0) == log(m_wet))
@constraint(model, c2[i = Dim], r[i](t0) == r_init[i])
@constraint(model, c3[i = Dim], rdot[i](t0) == rdot_init[i])
@constraint(model, c4[i = Dim], T[i](t0) == 0)

# Dynamics
@constraint(model, c5, @deriv(z, t) == -α*σ)
@constraint(model, c6[i = Dim], @deriv(r[i], t) == rdot[i])
@constraint(model, c7[i = Dim], @deriv(rdot[i], t) == T[i]/exp(z) + gm[i])

# Thrust Constraints
@constraint(model, c8, sqrt(sum((T[j]/exp(z))^2 for j in Dim)) <= σ)
@constraint(model, c9, ρ₁*exp(-log(m_wet - α*ρ₂*t))*(1 - (z-log(m_wet - α*ρ₂*t)) + (z-log(m_wet - α*ρ₂*t))^2/2) <= σ)
@constraint(model, c10, ρ₂*exp(-log(m_wet - α*ρ₂*t))*(1 - (z-log(m_wet - α*ρ₂*t))) >= σ)

# Mass Constraints
@constraint(model, c11, log(m_wet - α*ρ₂*t) <= z)
@constraint(model, c12, log(m_wet - α*ρ₁*t) >= z)

# SOCP Constraints
@constraint(model, c13, sqrt(sum((S*[r; rdot]).^2)) + c'*[r; rdot] <= 0)

# Final Conditions
@constraint(model, c14[i = Dim], r[i](tf) == r_final[i])
@constraint(model, c15[i = Dim], rdot[i](tf) == rdot_final[i])

# Solve the model
optimize!(model)

# Get the results
termination_status(model)
raw_status(model)
# primal_status(model)
# opt_obj = objective_value(model)
# r_opt = value.(r)
# rdot_opt = value.(rdot)
# T_opt = value.(T)
# # u_ts = supports.(u)

# # Plot the results
# plot(r_opt[1], r_opt[2], r_opt[3], label = "Trajectory", xlabel = "x (m)", ylabel = "y (m)", zlabel = "z (m)", legend = false)