#=Dependencies (MAY NEED TO RUN THIS)
using Pkg
Pkg.add("JuMP")
Pkg.add("SCS")
Pkg.add("Plots")
=#

#= Conventions - Wronggg - change to Ackmise
Coordiante system: [x,y,z] as defined by Figure 4 in https://www.researchgate.net/publication/335611189_Generalized_hp_Pseudospectral-Convex_Programming_for_Powered_Descent_and_Landing
=#

# Select Packages
using JuMP
using LinearAlgebra
import Ipopt
import Plots
import Interpolations

# Create JuMP Model - nonlinear solver
model = Model(optimizer_with_attributes(Ipopt.Optimizer))

# Define parameters
# Constant Parameters
const ge = [-9.807,0,0] # Earth's gravity
const gm = [-3.7114,0,0] # Mars' gravity
const maxThrust = 3100 # Maximum thrust (N)
const nThrusters = 6 # Number of thrusters 
const ϕ = deg2rad(27) # Cant angle of thrusters
const I_sp = 225 # Specific Impulse (s)
const m_wet = 2500 # 1905 # Wet mass of spacecraft (kg)
const m_dry = 1505 # Dry mass of spacecraft (kg)
const α = 1/(I_sp*abs(ge[1])*cos(ϕ)) # Constant for mass loss rate
const minGlide = deg2rad(86) # Minimum glide angle / Cone angle "90 - minGlide"

# SOCP Problem Formulation Stuff
S1 = [0; 1; 0; 0; 0; 0]
S2 = [0; 0; 1; 0; 0; 0]
c = [-tan(minGlide); 0; 0; 0; 0; 0]

# Time discretisation
const n = 5000 # Number of points (mesh size)
const tf = 72 # Time of flight (s)
const Δt = tf / n # Time step

# Initial Conditions
const initialX = 1500 # Initial altitude at descent (m)
const initialY = 0    # Initial ground track at descent (m)
const initialZ = 2000 # Initial ground track at descent (m)
const initialVx = -75 # Initial velocity (m/s)
const initialVy = 0 # Initial velocity (m/s)
const initialVz = 100 # Initial velocity (m/s)
const initialTx = 0 # Initial thrust x (N)
const initialTy = 0 # Initial thrust y (N)
const initialTz = 0 # Initial thrust z (N)

# Variable constraints 
const rho1 = 0.3 * maxThrust * nThrusters 
const rho2 = 0.8 * maxThrust * nThrusters

# Which variables do I need? Whcih variables need constraining?
# Mass, Thrust, Displacement (relative to target), Velocity
@variables(model, begin
    0 ≤ mass[1:n]  # Mass vector
    dispx[1:n] # 3Dof Displacement (relative to target) vector
    dispy[1:n]
    dispz[1:n]
    velx[1:n]  # 3Dof Velocity vector
    vely[1:n]
    velz[1:n]
    thrustx[1:n] # 3Dof Thrust vector
    thrusty[1:n]
    thrustz[1:n]
    # θ[1:n] <= minGlide # Glide angle
end)

# Fix initial conditions
# Inital mass = wet mass
fix(mass[1], m_wet; force = true)

# Inital displacement coordinates
fix(dispx[1], initialX; force = true) 
fix(dispy[1], initialY; force = true)
fix(dispz[1], initialZ; force = true)

# Intial velocity
fix(velx[1], initialVx; force = true)
fix(vely[1], initialVy; force = true)
fix(velz[1], initialVz; force = true)

# Fix final conditions
# Final displacement coordinates
fix(dispx[n], 0; force = true)
fix(dispy[n], 0; force = true)
fix(dispz[n], 0; force = true)

# Final velocity
fix(velx[n], 0; force = true)
fix(vely[n], 0; force = true)
fix(velz[n], 0; force = true)

#=
x_s = [m_wet, initialX, initialY, initialZ, initialVx, initialVy, initialVz, initialTx, initialTy, initialTz]
x_t = [m_wet, 0, 0, 0, 0, 0, 0, 0, 0, 0]
interp_linear = Interpolations.LinearInterpolation([1, n], [x_s, x_t])
initial_guess = mapreduce(transpose, vcat, interp_linear.(1:n))
set_start_value.(all_variables(model), vec(initial_guess))
=#

# Translational dynamics model
@expression(model, δm[j = 1:n], -α * (thrustx[j]^2 + thrusty[j]^2 + thrustz[j]^2)^0.5)

@expression(model, accx[j = 1:n],  thrustx[j]/mass[j] + gm[1])
@expression(model, accy[j = 1:n],  thrusty[j]/mass[j] + gm[2])
@expression(model, accz[j = 1:n],  thrustz[j]/mass[j] + gm[3])

# SOCP Constraints
# @expression(model, stateVector[1:6, j = 1:n], vcat(displacement[j,1:3], velocity[j,1:3]))
# @expression(model, c[1:6, j = 1:n], [-tan(minGlide); 0; 0; 0; 0; 0])
# @expression(model, SsV[1:2, j = 1:n], S*stateVector[1:6, j])
# @expression(model, ph[1:2, j = 1:n], S*stateVector[j, 1:6])
@expression(model, Sx1[j = 1:n], S1'*[dispx[j]; dispy[j]; dispz[j]; velx[j]; vely[j]; velz[j]])
@expression(model, Sx2[j = 1:n], S2'*[dispx[j]; dispy[j]; dispz[j]; velx[j]; vely[j]; velz[j]])

# Rotational dynamics model soon????

# NOTE: Only use constraints for variables defined above, otherwise use expression for secondary variables
for j = 2:n
    i = j-1
    @constraints(model, begin 
        # Dynamics model constraints
        mass[j] == mass[i] + (Δt * δm[i])
        velx[j] == velx[i] + (Δt * accx[i])
        vely[j] == vely[i] + (Δt * accy[i])
        velz[j] == velz[i] + (Δt * accz[i])
        dispx[j] == dispx[i] + (Δt * velx[i])
        dispy[j] == dispy[i] + (Δt * vely[i])
        dispz[j] == dispz[i] + (Δt * velz[i])
        # Actual constraints
        rho1 ≤ (thrustx[j]^2 + thrusty[j]^2 + thrustz[j]^2)^0.5 ≤ rho2
        dispx[j] ≥ 0
        # θ[j] == atan(sqrt(displacement[j,1]^2+displacement[j,2]^2)/displacement[j,3]^2)
        # SOCP constraint here 
        (Sx1[j]^2+Sx2[j]^2)^0.5 + c'*[dispx[j]; dispy[j]; dispz[j]; velx[j]; vely[j]; velz[j]] ≤ 0
        # [-c[1:6, j]*stateVector[j, 1:6]; S*stateVector[j, 1:6]] in SecondOrderCone()
        # (SsV[1,j]^2+SsV[2,j]^2)^0.5 + (c[1:6,j]')*stateVector[1:6,j] ≤ 0
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
        return Plots.plot(
            (1:n) * Δt,
            value.(y);
            xlabel = "Time (s)",
            legend = false,
            kwargs...,
        )
    end

# Plot results 
# Postion, Velocity, Acceleration, Net Force, Throttle Level, θ
plotter(dispx[:], label = "Altitude", linewidth = 2)
plotter(velx[:], label = "X-Velocity", linewidth = 2)
plotter(accx[:], label = "X-Acceleration", linewidth = 2)
plotter(thrustx[:], label = "X-Thrust", linewidth = 2)