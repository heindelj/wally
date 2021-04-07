include("water_tools.jl")
include("units.jl")

using Libdl
using Base.Filesystem

abstract type AbstractPotential end

# TTM struct and some useful constructors
mutable struct TTM <: AbstractPotential
    potential_function::Ptr
    full_lib_path::String
    version::AbstractString

    update_data::Bool
    current_energy::Float64
    current_gradients::Union{Array{Float64, 1}, Array{Float64, 2}}
end

TTM(full_lib_path::AbstractString) = TTM(dlsym(dlopen(full_lib_path), "ttm2f"), full_lib_path, "ttm2f", true, 0.0, zeros((0,0)))
TTM(full_lib_path::AbstractString, version::AbstractString) = TTM(dlsym(dlopen(full_lib_path), version), full_lib_path, version, true, 0.0, zeros((0,0)))
TTM() = TTM(dlsym(dlopen("/home/heindelj/Research/Sotiris/MBE_Dynamics/MBE_Dynamics_Home_Code/pyMD/bin/ttm_all.so"), "ttm2f"), 
            "/home/heindelj/Research/Sotiris/MBE_Dynamics/MBE_Dynamics_Home_Code/pyMD/bin/ttm_all.so", "ttm2f", true, 0.0, zeros((0,0)))
TTM(version::AbstractString) = TTM(dlsym(dlopen("/home/heindelj/Research/Sotiris/MBE_Dynamics/MBE_Dynamics_Home_Code/pyMD/bin/ttm_all.so"), version), 
            "/home/heindelj/Research/Sotiris/MBE_Dynamics/MBE_Dynamics_Home_Code/pyMD/bin/ttm_all.so", version, true, 0.0, zeros((0,0)))
TTM(ttm::TTM) = TTM(ttm.full_lib_path, ttm.version)

function get_energy_and_gradients(potential::TTM, coords::AbstractArray; sort_coords::Bool=true, reshape_coords::Bool=false)
    if reshape_coords
        coords = reshape(coords, (3, div(length(coords), 3)))
    end
    num_waters::Int32 = size(coords, 2) / 3
    grads = zero(coords)
    energy = Float64[0]
    if sort_coords
        new_coords = sort_water_molecules_to_oxygens_first(coords)
        ccall(potential.potential_function, Cvoid, (Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Float64}), num_waters, new_coords, grads, energy)
    else
        ccall(potential.potential_function, Cvoid, (Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Float64}), num_waters, coords, grads, energy)
    end
    if sort_coords
        grads = sort_oxygens_first_to_water_molecules(grads)
    end
    if reshape_coords
        grads = vec(grads)
    end
    return energy[begin] * conversion(:kcal, :hartree), grads * conversion(:kcal, :hartree)
end

function get_energy(potential::TTM, coords::AbstractArray; sort_coords::Bool=true, reshape_coords::Bool=false)
    if potential.update_data
        cwd = String(pwd())
        cd(dirname(potential.full_lib_path))
        potential.current_energy, potential.current_gradients = get_energy_and_gradients(potential, coords, sort_coords=sort_coords, reshape_coords=reshape_coords)
        cd(cwd)
        potential.update_data = false
        return potential.current_energy
    else
        potential.update_data = true
        return potential.current_energy
    end
end

function get_gradients(potential::TTM, coords::AbstractArray; sort_coords::Bool=true, reshape_coords::Bool=false)
    if potential.update_data
        cwd = String(pwd())
        cd(dirname(potential.full_lib_path))
        potential.current_energy, potential.current_gradients = get_energy_and_gradients(potential, coords, sort_coords=sort_coords, reshape_coords=reshape_coords)
        cd(cwd)
        potential.update_data = false
        return potential.current_gradients
    else
        potential.update_data = true
        return potential.current_gradients
    end
end

function get_gradients!(potential::TTM, storage::AbstractArray, coords::AbstractArray; sort_coords::Bool=true, reshape_coords::Bool=false)
    storage[:] = get_gradients(potential, coords, sort_coords=sort_coords, reshape_coords=reshape_coords)
end

### MBPol Water Potential ###

mutable struct MBPol <: AbstractPotential
    potential_function::Ptr
    full_lib_path::String
    update_data::Bool
    current_energy::Float64
    current_gradients::Union{Array{Float64, 1}, Array{Float64, 2}}
end
MBPol(full_lib_path::AbstractString) = MBPol(dlsym(dlopen(full_lib_path), "calcpotg_"), full_lib_path, true, 0.0, zeros((0,0)))
MBPol(mbpol::MBPol) = MBPol(mbpol.full_lib_path)

function get_energy_and_gradients(potential::MBPol, coords::AbstractArray; reshape_coords::Bool=false)
    if reshape_coords
        coords = reshape(coords, (3, div(length(coords), 3)))
    end
    num_waters::Int32 = size(coords, 2) / 3
    grads = zero(coords)
    energy = Float64[0]

    # call the potential
    ccall(potential.potential_function, Cvoid, (Ref{Cint}, Ref{Float64}, Ref{Float64}, Ref{Float64}), num_waters, energy, coords, grads)
    if reshape_coords
        grads = vec(grads)
    end
    return energy[begin] / 627.5, grads / 627.5
end

function get_energy(potential::MBPol, coords::AbstractArray; reshape_coords::Bool=false)
    if potential.update_data
        cwd = String(pwd())
        cd(dirname(potential.full_lib_path))
        potential.current_energy, potential.current_gradients = get_energy_and_gradients(potential, coords, reshape_coords=reshape_coords)
        cd(cwd)
        potential.update_data = false
        return potential.current_energy
    else
        potential.update_data = true
        return potential.current_energy
    end
end

function get_gradients(potential::MBPol, coords::AbstractArray; reshape_coords::Bool=false)
    if potential.update_data
        cwd = String(pwd())
        cd(dirname(potential.full_lib_path))
        potential.current_energy, potential.current_gradients = get_energy_and_gradients(potential, coords, reshape_coords=reshape_coords)
        cd(cwd)
        potential.update_data = false
        return potential.current_gradients
    else
        potential.update_data = true
        return potential.current_gradients
    end
end

function get_gradients!(potential::MBPol, storage::AbstractArray, coords::AbstractArray; reshape_coords::Bool=false)
    storage[:] = get_gradients(potential, coords, reshape_coords=reshape_coords)
end

### PROTONATED WATER POTENTIAL ###

mutable struct Protonated_Water <: AbstractPotential
    full_lib_path::AbstractString
    init_function::Ptr
    energy_function::Ptr
    energy_and_gradient_function::Ptr
    is_initialized::Int32 # for compatibility with fortran
end

Protonated_Water(full_lib_path::AbstractString) = Protonated_Water(full_lib_path, 
dlsym(dlopen(full_lib_path), "initialize_potential_"),
dlsym(dlopen(full_lib_path), "get_energy_"),
dlsym(dlopen(full_lib_path), "get_energy_and_gradients_"),
0)

Protonated_Water() = Protonated_Water("/home/heindelj/Research/Sotiris/Potentials/PES_HP/PES/libBowmanProtWater.so")

function get_energy(potential::Protonated_Water, coords::AbstractArray; reshape_coords::Bool=false)
    num_waters::Int32 = 0
    if reshape_coords
        coords = reshape(coords, (3, :))
    end
    num_waters = get_num_waters(coords)

    cwd = String(pwd())
    cd(dirname(potential.full_lib_path))
    if potential.is_initialized == 0
        ccall(potential.init_function, Cvoid, (Ref{Int32},), num_waters)
        potential.is_initialized = 1
    end
    energy = Float64[0]
    ccall(potential.energy_function, Cvoid, (Ref{Cdouble}, Ref{Int32}, Ref{Cdouble}), coords, num_waters, energy)
    cd(cwd)

    if reshape_coords
        coords = vec(coords)
    end

    return energy[begin]
end

function get_energy_and_gradients(potential::Protonated_Water, coords::AbstractArray; reshape_coords::Bool=false)
    num_waters::Int32 = 0
    if reshape_coords
        coords = reshape(coords, (3, :))
    end
    num_waters = get_num_waters(coords)

        cwd = String(pwd())
        cd(dirname(potential.full_lib_path))

    if potential.is_initialized == 0
        ccall(potential.init_function, Cvoid, (Ref{Int32},), num_waters)
        potential.is_initialized = 1
    end

    energy = Float64[0]
    grads = zeros(Float64, length(coords))
    ccall(potential.energy_and_gradient_function, Cvoid, (Ref{Cdouble}, Ref{Int32}, Ref{Cdouble}, Ref{Cdouble}), coords, num_waters, energy, grads)
    cd(cwd)

    if reshape_coords
        coords = vec(coords)
        grads = vec(grads)
    else
        grads = reshape(grads, size(coords))
    end
    return energy[begin], grads
end

function get_gradients(potential::Protonated_Water, coords::AbstractArray; reshape_coords::Bool=false)
    return get_energy_and_gradients(potential, coords, reshape_coords=reshape_coords)[2]
end

function get_gradients!(potential::Protonated_Water, storage::AbstractArray, coords::AbstractArray; reshape_coords::Bool=false)
    storage[:] = get_energy_and_gradients(potential, coords, reshape_coords=reshape_coords)[2]
end

function get_num_waters(coords::AbstractArray)
    num_waters::Int32 = (size(coords, 2) - 4)
    if num_waters < 0
        return Int32(1)
    else
        return num_waters / 3
    end
end