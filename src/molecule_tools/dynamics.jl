include("call_potential.jl")
include("read_xyz.jl")

using Distributions, Random

abstract type AbstractIntegrator end
abstract type AbstractSimulation end

mutable struct Simulation <: AbstractSimulation
    nsteps::Int
    integrator::AbstractIntegrator
    coords::Matrix{Float64}
    velocities::Matrix{Float64}
    forces::Matrix{Float64}
    labels::Vector{String}
    masses::Vector{Float64}
    potential::AbstractPotential
    ofile::String
    write_every::Int
end

function run_simulation!(sim::Simulation)
    for i_step in 1:sim.nsteps
        energy = take_step!(sim.integrator, sim.coords, sim.velocities,
                   sim.forces, sim.masses, sim.potential)
        if (i_step - 1) % sim.write_every == 0
            if i_step == 1
                write_xyz(sim.ofile, [string(length(sim.masses), "\nV = ", energy, " E_kin = ", get_kinetic_energy(sim.velocities, sim.masses), " T = ", get_temperature(sim.velocities, sim.masses))], [sim.labels], [sim.coords])
            else
                write_xyz(sim.ofile, [string(length(sim.masses), "\nV = ", energy, " E_kin = ", get_kinetic_energy(sim.velocities, sim.masses), " T = ", get_temperature(sim.velocities, sim.masses) * 315775.0248)], [sim.labels], [sim.coords], append=true)
            end
        end
    end
end

struct VelocityVerlet <: AbstractIntegrator
    dt::Float64
end

struct AndersenThermostat <: AbstractIntegrator
    dt::Float64
    T::Float64
    AndersenThermostat(dt::Float64, T::Float64) = new(dt, T / 315775.0248)
end

struct LangevinThermostat <: AbstractIntegrator
    dt::Float64
    T::Float64
    α::Float64

    a::Vector{Float64}
    sqrtb::Vector{Float64}
    σ::Float64
    generator::Normal{Float64}
    gaussian_noise::Matrix{Float64}
end
LangevinThermostat(dt::Float64, T::Float64, α::Float64, masses::Vector{Float64}) = LangevinThermostat(dt, T / 315775.0248, α, (1.0 .- 0.5 * α * dt ./ masses) ./ (1.0 .+ 0.5 * α * dt ./ masses), sqrt.(1.0 ./ (1.0 .+ 0.5 * α * dt ./ masses)), sqrt(2.0*α*T*dt), Normal(0.0, sqrt(2.0*α*T*dt)), rand(Normal(0.0, sqrt(2.0*α*T*dt)), (3, length(masses))))

function take_step!(integrator::LangevinThermostat, coords::Matrix{Float64}, velocities::Matrix{Float64}, forces::Matrix{Float64}, masses::Vector{Float64}, potential::AbstractPotential)
    """
    Moves the coordinates according to velocity verlet integration
    using the provided velocities and forces.
    """
    @views begin
        old_gaussian_noise = copy(integrator.gaussian_noise)
        integrator.gaussian_noise[:,:] = rand(integrator.generator, size(coords))
        velocities[:,:] = (integrator.a' .* velocities .+
                           integrator.sqrtb' .*
                           get_accelerations(forces, masses) * integrator.dt .+
                           (integrator.sqrtb ./ (2 * masses))' .*
                           (integrator.gaussian_noise + old_gaussian_noise))
        coords[:,:] += integrator.sqrtb' .* velocities * integrator.dt * conversion(:bohr, :angstrom)
        energy, grads = get_energy_and_gradients(potential, coords)
        forces[:,:] = -grads
        return energy
    end
end

# function take_step!(integrator::LangevinThermostat, coords::Matrix{Float64}, velocities::Matrix{Float64}, forces::Matrix{Float64}, masses::Vector{Float64}, potential::AbstractPotential)
#     """
#     Moves the coordinates according to velocity verlet integration
#     using the provided velocities and forces.
#     """
#     @views begin
#         velocities[:,:] = (integrator.a' .* velocities .+
#                            get_accelerations(forces, masses) * integrator.dt .+
#                            (integrator.sqrtb.^2 ./ masses)' .*
#                            integrator.gaussian_noise)
#         integrator.gaussian_noise[:,:] = rand(integrator.generator, size(coords))
#         coords[:,:] += integrator.sqrtb'.^2 .* 
#         (velocities * integrator.dt .+
#         (integrator.dt ./ (2 * masses))' .* integrator.gaussian_noise)
#         energy, grads = get_energy_and_gradients(potential, coords)
#         forces[:,:] = -grads
#         return energy
#     end
# end

function get_kinetic_energy(vels::Matrix{Float64}, masses::Vector{Float64})
    ke = 0.0
    for i in 1:length(masses)
        @views ke += 0.5 * masses[i] * dot(vels[:,i], vels[:,i])
    end
    return ke
end

function get_temperature(vels::Matrix{Float64}, masses::Vector{Float64})
    return 2.0 * get_kinetic_energy(vels, masses) / (3 * length(masses))
end

function get_accelerations(forces::Matrix{Float64}, masses::Vector{Float64})
    return hcat([forces[:,i] / masses[i] for i in 1:length(masses)]...)
end

function maxwell_boltzmann!(velocities::Matrix{Float64}, masses::Vector{Float64}, T::Float64)
    sigmas = sqrt.(T ./ masses)
    for i in 1:length(sigmas)
        @views velocities[:,i] = rand(Normal{Float64}(0.0, sigmas[i]), size(velocities, 1))
    end
end

function take_step!(integrator::AndersenThermostat, coords::Matrix{Float64}, velocities::Matrix{Float64}, forces::Matrix{Float64}, masses::Vector{Float64}, potential::AbstractPotential)
    @views begin
        maxwell_boltzmann!(velocities, masses, integrator.T)
        coords[:,:] += velocities * integrator.dt + 0.5 * get_accelerations(forces, masses) * integrator.dt^2
        energy, grads = get_energy_and_gradients(potential, coords)
        forces[:,:] = -grads # above function returns gradients, so switch to forces
        return energy
    end
end

function take_step!(integrator::VelocityVerlet, coords::Matrix{Float64}, velocities::Matrix{Float64}, forces::Matrix{Float64}, masses::Vector{Float64}, potential::AbstractPotential)
    """
    Moves the coordinates according to velocity verlet integration
    using the provided velocities and forces.
    """
    @views begin
        coords[:,:] += velocities * integrator.dt + 0.5 * get_accelerations(forces, masses) * integrator.dt^2
        energy, grads = get_energy_and_gradients(potential, coords)
        velocities[:,:] += 0.5 * get_accelerations(forces, masses) * integrator.dt
        forces[:,:] = -grads # above function returns gradients, so switch to forces
        velocities[:,:] += 0.5 * get_accelerations(forces, masses) * integrator.dt
        return energy
    end
end

function take_step!(integrator::LangevinThermostat, coords::Matrix{Float64}, velocities::Matrix{Float64}, forces::Matrix{Float64}, masses::Vector{Float64}, potential::AbstractPotential)
    """
    Moves the coordinates according to velocity verlet integration
    using the provided velocities and forces.
    """
    @views begin
        coords[:,:] += velocities * integrator.dt + 0.5 * get_accelerations(forces, masses) * integrator.dt^2
        energy, grads = get_energy_and_gradients(potential, coords)
        velocities[:,:] += 0.5 * get_accelerations(forces, masses) * integrator.dt
        forces[:,:] = -grads # above function returns gradients, so switch to forces
        velocities[:,:] += 0.5 * get_accelerations(forces, masses) * integrator.dt
        return energy
    end
end


