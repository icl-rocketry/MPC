#=Dependencies (MAY NEED TO RUN THIS)
using Pkg
Pkg.add("JuMP")
Pkg.add("SCS")
Pkg.add("Plots")
=#

#= Conventions
Coordiante system: [x,y,z] as defined by Figure 4 in https://www.researchgate.net/publication/335611189_Generalized_hp_Pseudospectral-Convex_Programming_for_Powered_Descent_and_Landing
=#

# Select Packages
using JuMP
using LinearAlgebra
import Ipopt
import Plots

# Create JuMP Model - nonlinear solver
model = Model(optimizer_with_attributes(Ipopt.Optimizer))

# Define parameters
# Constant Parameters
const ge = [-9.807,0,0] # Earth's gravity
const gm = [-3.7114,0,0] # Mars' gravity
const maxThrust = 3100 # Maximum thrust (N)
const nThrusters = 6 # Number of thrusters 
const ϕ = 27 # Cant angle of thrusters
const I_sp = 225 # Specific Impulse (s)
const m_wet = 1905 # Wet mass of spacecraft (kg)
const m_dry = 1505 # Dry mass of spacecraft (kg)
const α = 1/(I_sp*abs(ge[3])*cosd(ϕ)) # Constant for mass loss rate
const minGlide = deg2rad(86) # Minimum glide angle / Cone angle "90 - minGlide"

# SOCP Problem Formulation Stuff
const S = [0 1 0 0 0 0; 0 0 1 0 0 0]

# Time discretisation
const n = 1000 # Number of points (mesh size)
const tf = 72 # Time of flight (s)
const Δt = tf / n # Time step

# Initial Conditions
const initialX = 1500 # Initial altitude at descent (m)
const initialY = 0    # Initial ground track at descent (m)
const initialZ = 2000 # Initial ground track at descent (m)
const initialVx = -75 # Initial velocity (m/s)
const initialVy = 0 # Initial velocity (m/s)
const initialVz = 100 # Initial velocity (m/s)

# Variable constraints 
const rho1 = 0.3 * maxThrust * nThrusters 
const rho2 = 0.8 * maxThrust * nThrusters

# Which variables do I need? Whcih variables need constraining?
# Mass, Thrust, Displacement (relative to target), Velocity
@variables(model, begin
    0 <= mass[1:n]  # Mass vector
    displacement[1:n, 1:3] # 3Dof Displacement (relative to target) vector
    velocity[1:n, 1:3]  # 3Dof Velocity vector
    thrust[1:n, 1:3] # 3Dof Thrust vector
    θ[1:n] <= minGlide # Glide angle
end)

# Fix initial conditions
# Inital mass = wet mass
fix(mass[1], m_wet; force = true)

# Inital displacement coordinates
fix(displacement[1,1], initialX; force = true) 
fix(displacement[1,2], initialY; force = true)
fix(displacement[1,3], initialZ; force = true)

# Intial velocity
fix(velocity[1,1], initialVx; force = true)
fix(velocity[1,2], initialVy; force = true)
fix(velocity[1,3], initialVz; force = true)

# Fix final conditions
# Final displacement coordinates
fix(displacement[n,1], 0; force = true)
fix(displacement[n,2], 0; force = true)
fix(displacement[n,3], 0; force = true)

# Final velocity
fix(velocity[n,1], 0; force = true)
fix(velocity[n,2], 0; force = true)
fix(velocity[n,3], 0; force = true)

# Translational dynamics model
@expression(model, δm[j = 1:n], -α * (thrust[j,1]^2 + thrust[j,2]^2 + thrust[j,3]^2)^0.5)
@expression(model, acceleration[j = 1:n, 1:3], thrust[j,1:3]./mass[j] .+ gm)

# SOCP Constraints
@expression(model, stateVector[1:6, j = 1:n], vcat(displacement[j,1:3], velocity[j,1:3]))
@expression(model, c[1:6, j = 1:n], [-tan(minGlide); 0; 0; 0; 0; 0])
@expression(model, SsV[1:2, j = 1:n], S*stateVector[1:6, j])
# @expression(model, ph[1:2, j = 1:n], S*stateVector[j, 1:6])

# Rotational dynamics model soon????

# NOTE: Only use constraints for variables defined above, otherwise use expression for secondary variables
for j = 2:n
    i = j-1
    @constraints(model, begin 
        # Dynamics model constraints
        mass[j] == mass[j-1] + Δt * δm[j-1]
        velocity[j, 1:3] == velocity[i, 1] .+ (Δt .* acceleration[i,1:3])
        displacement[j, 1:3] == displacement[i, 1:3] .+ (Δt .* velocity[i,1:3])
        # Actual constraints
        rho1 ≤ (thrust[j,1]^2 + thrust[j,2]^2 + thrust[j,3]^2)^0.5 ≤ rho2
        displacement[j,1] ≥ 0
        # θ[j] == atan(sqrt(displacement[j,1]^2+displacement[j,2]^2)/displacement[j,3]^2)
        # SOCP constraint here 
        # [-c[1:6, j]*stateVector[j, 1:6]; S*stateVector[j, 1:6]] in SecondOrderCone()
        (SsV[1,j]^2+SsV[2,j]^2)^0.5 + (c[1:6,j]')*stateVector[1:6,j] ≤ 0
    end)
end

# Set Objective: Minimize fuel usage/Maximise final mass
@objective(model, Max, mass[n])

# Run solver
optimize!(model)
solution_summary(model)


# FUNCTIONS
function plotter(y; kwargs...)
    """
    Quick Plotter
    """
        return plot(
            (1:n) * Δt,
            value.(y);
            xlabel = "Time (s)",
            legend = false,
            kwargs...,
        )
    end

# Plot results 
# Postion, Velocity, Acceleration, Net Force, Throttle Level, θ
plotter(displacement[:,3], label = "Altitude", linewidth = 2)
plotter(velocity[:,3], label = "Z-Velocity", linewidth = 2)
plotter(acceleration[:,3], label = "Z-Acceleration", linewidth = 2)
plotter(thrust[:,3], label = "Z-Thrust", linewidth = 2)