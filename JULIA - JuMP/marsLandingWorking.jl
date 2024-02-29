#=Dependencies (MAY NEED TO RUN THIS)
using Pkg
Pkg.add("JuMP")
Pkg.add("SCS")
Pkg.add("Plots")
Pkg.add("PlotThemes")
=#

#= Conventions
Coordiante system: [x,y,z] as defined by Figure 4 in https://www.researchgate.net/publication/335611189_Generalized_hp_Pseudospectral-Convex_Programming_for_Powered_Descent_and_Landing
=#

# Select Packages
using JuMP
using LinearAlgebra
import Ipopt
import Plots

# Plot settings
Plots.theme(:orange)

# Create JuMP Model - nonlinear solver
model = Model(optimizer_with_attributes(Ipopt.Optimizer))

# Define parameters
# Constant Parameters
const ge = [-9.807,0,0] # Earth's gravity
const gm = [-3.7114,0,0] # Mars' gravity
const maxThrust = 500 # Maximum thrust (N)
const nThrusters = 6 # Number of thrusters 
const ϕ = 27 # Cant angle of thrusters
const I_sp = 225 # Specific Impulse (s)
const m_wet = 1905 # Wet mass of spacecraft (kg)
const m_dry = 1505 # Dry mass of spacecraft (kg)
const α = 1/(I_sp*abs(gm[1])*cosd(ϕ)) # Constant for mass loss rate
const minGlide = deg2rad(4) # Minimum glide angle / Cone angle "90 - minGlide"

# Time discretisation
const n = 100 # Number of points (mesh size)
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
const rho1 = 0.3 * maxThrust * nThrusters * cosd(ϕ)
const rho2 = 0.8 * maxThrust * nThrusters * cosd(ϕ)

# Define variables
@variables(model, begin
    # Decision Variables
    ux[1:n]
    uy[1:n]
    uz[1:n]
    σ[1:n]
    # Other variables
    rx[1:n]
    ry[1:n]
    rz[1:n]
    vx[1:n]
    vy[1:n]
    vz[1:n]
    z[1:n]
    z0[1:n]
end)

# Boundary Conditions
fix(z[1], log(m_wet); force = true)
fix(z0[1], log(m_wet); force = true)

fix(rx[1], initialX; force = true) 
fix(ry[1], initialY; force = true)
fix(rz[1], initialZ; force = true)

fix(vx[1], initialVx; force = true)
fix(vy[1], initialVy; force = true)
fix(vz[1], initialVz; force = true)

#=
fix(rx[n], 0; force = true)
fix(ry[n], 0; force = true)
fix(rz[n], 0; force = true)

fix(vx[n], 0; force = true)
fix(vy[n], 0; force = true)
fix(vz[n], 0; force = true)
=#

# Translational dynamics model
@expression(model, μ1[j=1:n], rho1*exp(-z0[j]))
@expression(model, μ2[j=1:n], rho2*exp(-z0[j]))

# NOTE: Only use constraints for variables defined above, otherwise use expression for secondary variables
for j = 2:n
    i = j-1
    @constraints(model, begin 
    # fcn2optimexpr
        z0[j] == log(m_wet - α * rho2 * Δt * i)
        z[j] == z[i] - (-α * (σ[j] + σ[i])/2) * Δt/2
        rx[j] == rx[i] + (vx[j] + vx[i]) * Δt/2 + (ux[j] + ux[i]) * Δt^2/12
        ry[j] == ry[i] + (vy[j] + vy[i]) * Δt/2 + (uy[j] + uy[i]) * Δt^2/12
        rz[j] == rz[i] + (vz[j] + vz[i]) * Δt/2 + (uz[j] + uz[i]) * Δt^2/12
        vx[j] == vx[i] + ((ux[j] + ux[i])/2 + gm[1]) * Δt/2
        vy[j] == vy[i] + ((uy[j] + uy[i])/2 + gm[2]) * Δt/2
        vz[j] == vz[i] + ((uz[j] + uz[i])/2 + gm[3]) * Δt/2

    # Actual Constraints
        rx[j] ≥ 0
        μ1[j]*(1 - (z[j]-z0[j]) + (z[j]-z0[j])^2/2) ≤ σ[j]
        σ[j] ≤ μ2[j]*(1 - (z[j]-z0[j]))
        z0[j] ≤ z[j]
        z[j] ≤ log(m_wet - α*rho1*Δt*i)
        0 ≤ σ[i]^2 - (ux[i]^2 + uy[i]^2 + uz[i]^2)
        rho1^2 ≤ (ux[i]^2 + uy[i]^2 + uz[i]^2) * exp(z[i])^2
        (ux[i]^2 + uy[i]^2 + uz[i]^2) * exp(z[i])^2 ≤ rho2^2
        (ry[i]^2 + rz[i]^2) ≤ (rx[i]*tan(minGlide))^2
    end)
end

#=
for k = 1:n
    @constraints(model, begin
    0 ≤ σ[k]^2 - (ux[k]^2 + uy[k]^2 + uz[k]^2)
    rho1^2 ≤ (ux[k]^2 + uy[k]^2 + uz[k]^2) * exp(z[k])^2
    (ux[k]^2 + uy[k]^2 + uz[k]^2) * exp(z[k])^2 ≤ rho2^2
    (ry[k]^2 + rz[k]^2) ≤ (rx[k]*tan(minGlide))^2
    end)
end
=#

# PROBLEM: IPOPT DOES NOT LIKE L2-NORMS
# SOLUTION: SQUARE BOTH SIDES OF THE CONSTRAINTS

# PROBLEM: Dynamics (@ j = 2) do not make physical sense - need to check iteration, should not separate into 2 blocks?
# SOLUTION: ???

# Set Objective: Minimize fuel usage/Maximise final mass
@objective(model, Min, -z[n])

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
x1 = plotter(rx[:], ylabel = "X-Disp", linewidth = 2)
x2 = plotter(ry[:], ylabel = "Y-Disp", linewidth = 2)
x3 = plotter(rz[:], ylabel = "Z-Disp", linewidth = 2)
v1 = plotter(vx[:], ylabel = "X-Velocity", linewidth = 2)
v2 = plotter(vy[:], ylabel = "Y-Velocity", linewidth = 2)
v3 = plotter(vz[:], ylabel = "Z-Velocity", linewidth = 2)
u1 = plotter(ux[:], ylabel = "X-Acceleration", linewidth = 2)
u2 = plotter(uy[:], ylabel = "Y`-Acceleration", linewidth = 2)
u3 = plotter(uz[:], ylabel = "Z-Acceleration", linewidth = 2)
Plots.plot(x1, x2, x3, 
        v1, v2, v3, 
        u1, u2, u3,
        layout = (3,3), legend = false)

       