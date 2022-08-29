include("water_tools.jl")
include("units.jl")
include("read_xyz.jl")

using Libdl
using Base.Filesystem
using LinearAlgebra

abstract type AbstractPotential end

struct MultiPotential <: AbstractPotential
    potentials::Vector{AbstractPotential}
end

function get_energy_and_gradients(mp::MultiPotential, coords::AbstractMatrix)
    energy = 0.0
    grads = zero(coords)
    for potential in mp.potentials
        energy_temp, grads_temp = get_energy_and_gradients(potential, coords)
        energy += energy_temp
        @views grads[:,:] += grads_temp
    end
    return energy, grads
end

struct WCACubicConfinement <: AbstractPotential
    # should also include a center, but for now just assume centered at origin
    # Using a cube instead of sphere because the forces are easier to calculate.
    side_length::Float64
    σ::Float64
    ϵ::Float64
end


function get_energy_and_gradients(potential::WCACubicConfinement, coords::AbstractMatrix)
    grads = zero(coords)
    energy = 0.0
    for i in 1:size(coords, 2)
        for w in 1:size(coords, 1)
            if coords[w, i] > potential.side_length
                dist_past_box = coords[w, i] - potential.side_length
                σ_over_r_6 = (potential.σ / (dist_past_box-2^(1/6)*potential.σ))^6
                energy += 4.0 * potential.ϵ * (σ_over_r_6^2 - σ_over_r_6) + potential.ϵ
                g_iw = -24.0 * potential.ϵ * (2.0 * σ_over_r_6^2 - σ_over_r_6) / (dist_past_box - 2^(1/6) * potential.σ)
                grads[w,i] += g_iw
            elseif coords[w, i] < -potential.side_length
                dist_past_box = -potential.side_length - coords[w, i]
                σ_over_r_6 = (potential.σ / (dist_past_box-2^(1/6)*potential.σ))^6
                energy += 4.0 * potential.ϵ * (σ_over_r_6^2 - σ_over_r_6) + potential.ϵ
                g_iw = -24.0 * potential.ϵ * (2.0 * σ_over_r_6^2 - σ_over_r_6) / (dist_past_box - 2^(1/6) * potential.σ)
                grads[w,i] -= g_iw
            end
        end
    end
    return energy, grads
end

struct LennardJones <: AbstractPotential
    σ::Float64
    ϵ::Float64
end

function get_energy_and_gradients(potential::LennardJones, coords::AbstractMatrix)
    grads = zero(coords)
    energy = 0.0
    for i in 1:(size(coords, 2)-1)
        for j in (i+1):size(coords, 2)
            @views r_ij = coords[:,i] - coords[:,j]
            σ_over_r_6 = (potential.σ / norm(r_ij))^6
            energy += 4.0 * potential.ϵ * (σ_over_r_6^2 - σ_over_r_6)
            g_ij = -24.0 * potential.ϵ * (2.0 * σ_over_r_6^2 - σ_over_r_6) / norm(r_ij)^2 * r_ij
            @views grads[:,i] +=  g_ij
            @views grads[:,j] -=  g_ij
        end
    end
    return energy, grads
end

function get_energy(potential::LennardJones, coords::AbstractMatrix)
    energy = 0.0
    for i in 1:(size(coords, 2)-1)
        for j in (i+1):size(coords, 2)
            @views r_ij = coords[:,i] - coords[:,j]
            σ_over_r_6 = (potential.σ / norm(r_ij))^6
            energy += 4.0 * potential.ϵ * (σ_over_r_6^2 - σ_over_r_6)
        end
    end
    return energy
end

function finite_difference(potential::AbstractPotential, coords::Matrix{Float64}, step_size=1e-5)
    grads = zero(coords)
    for i in 1:size(coords, 2)
        for j in 1:size(coords,1)
            coords[j,i] += step_size
            f_plus_h = get_energy(potential, coords)
            coords[j,i] -= 2 * step_size
            f_minus_h = get_energy(potential, coords)
            coords[j,i] += step_size

            grads[j,i] = (f_plus_h - f_minus_h) / (2 * step_size)
        end
    end
    return grads
end

########################
###    TTM Water     ###
########################

mutable struct TTM <: AbstractPotential
    potential_function::Ptr
    full_lib_path::String
    version::String

    update_energy::Bool
    update_gradients::Bool
    current_energy::Float64
    current_gradients::Union{Array{Float64, 1}, Array{Float64, 2}, HybridArray{Tuple{3,StaticArrays.Dynamic()},Float64,2,2,Array{Float64,2}}, Nothing}
end

TTM(full_lib_path::AbstractString, version::AbstractString) = TTM(dlsym(dlopen(full_lib_path), version), full_lib_path, version, true, true, 0.0, nothing)
TTM(full_lib_path::AbstractString) = TTM(full_lib_path, "ttm2f")
TTM() = TTM("/home/heindelj/research/Sotiris/MBE_Dynamics/MBE_Dynamics_Home_Code/pyMD/bin/ttm_all.so")
TTM(ttm::TTM) = TTM(ttm.full_lib_path, ttm.version)

function get_energy_and_gradients(potential::TTM, coords::AbstractArray; sort_coords::Bool=true, reshape_coords::Bool=false)
    if reshape_coords
        coords = reshape(coords, (3, div(length(coords), 3)))
    end
    num_waters::Int32 = size(coords, 2) / 3
    grads = zero(coords)
    energy = Float64[0]
    if sort_coords
        new_coords = zero(coords)
        sort_water_molecules_to_oxygens_first!(new_coords, coords)
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
    if potential.update_energy
        cwd = String(pwd())
        cd(dirname(potential.full_lib_path))
        potential.current_energy, potential.current_gradients = get_energy_and_gradients(potential, coords, sort_coords=sort_coords, reshape_coords=reshape_coords)
        cd(cwd)
        potential.update_energy = true
        potential.update_gradients = false
        return potential.current_energy
    else
        potential.update_energy = true
        potential.update_gradients = true
        return potential.current_energy
    end
end

function get_gradients(potential::TTM, coords::AbstractArray; sort_coords::Bool=true, reshape_coords::Bool=false)
    if potential.update_gradients
        cwd = String(pwd())
        cd(dirname(potential.full_lib_path))
        potential.current_energy, potential.current_gradients = get_energy_and_gradients(potential, coords, sort_coords=sort_coords, reshape_coords=reshape_coords)
        cd(cwd)
        potential.update_gradients = true
        potential.update_energy = false
        return potential.current_gradients
    else
        potential.update_energy = true
        potential.update_gradients = true
        return potential.current_gradients
    end
end

function get_gradients!(potential::TTM, storage::AbstractArray, coords::AbstractArray; sort_coords::Bool=true, reshape_coords::Bool=false)
    storage[:] = get_gradients(potential, coords, sort_coords=sort_coords, reshape_coords=reshape_coords)
end

########################
###   MB-Pol Water   ###
########################

mutable struct MBPol <: AbstractPotential
    potential_function::Ptr
    full_lib_path::String
    update_energy::Bool
    update_gradients::Bool
    current_energy::Float64
    current_gradients::Union{Array{Float64, 1}, Array{Float64, 2}}
end
MBPol(full_lib_path::AbstractString) = MBPol(dlsym(dlopen(full_lib_path), "calcpotg_"), full_lib_path, true, true, 0.0, zeros((0,0)))
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
    return energy[begin] * conversion(:kcal, :hartree), grads * conversion(:kcal, :hartree)
end

function get_energy(potential::MBPol, coords::AbstractArray; reshape_coords::Bool=false)
    if potential.update_energy
        cwd = String(pwd())
        cd(dirname(potential.full_lib_path))
        potential.current_energy, potential.current_gradients = get_energy_and_gradients(potential, coords, reshape_coords=reshape_coords)
        cd(cwd)
        potential.update_energy = true
        potential.update_gradients = false
        return potential.current_energy
    else
        potential.update_energy = true
        potential.update_gradients = true
        return potential.current_energy
    end
end

function get_gradients(potential::MBPol, coords::AbstractArray; reshape_coords::Bool=false)
    if potential.update_gradients
        cwd = String(pwd())
        cd(dirname(potential.full_lib_path))
        potential.current_energy, potential.current_gradients = get_energy_and_gradients(potential, coords, reshape_coords=reshape_coords)
        cd(cwd)
        potential.update_energy = false
        potential.update_gradients = true
        return potential.current_gradients
    else
        potential.update_energy = true
        potential.update_gradients = true
        return potential.current_gradients
    end
end

function get_gradients!(potential::MBPol, storage::AbstractArray, coords::AbstractArray; reshape_coords::Bool=false)
    storage[:] = get_gradients(potential, coords, reshape_coords=reshape_coords)
end

########################
### Protonated Water ###
########################

mutable struct Protonated_Water <: AbstractPotential
    full_lib_path::String
    energy_function::Ptr
    energy_and_gradient_function::Ptr
    is_initialized::Int32 # for compatibility with fortran
end

Protonated_Water(full_lib_path::String) = Protonated_Water(full_lib_path,
dlsym(dlopen(full_lib_path), "get_energy"),
dlsym(dlopen(full_lib_path), "get_energy_and_gradients"),
0)

Protonated_Water() = Protonated_Water("/home/heindelj/research/Sotiris/Potentials/PES_HP/PES/libBowmanProtWater.so")
Protonated_Water(pwat::Protonated_Water) = Protonated_Water(pwat.full_lib_path)

function get_energy(potential::Protonated_Water, coords::AbstractArray; reshape_coords::Bool=false)
    if reshape_coords
        coords = reshape(coords, (3, :))
    end
    num_atoms::Int32 = size(coords, 2)

    cwd = String(pwd())
    cd(dirname(potential.full_lib_path))
    
    energy = Float64[0]
    ccall(potential.energy_function, Cvoid, (Ref{Cdouble}, Ref{Int32}, Ref{Int32}, Ref{Cdouble}), coords, potential.is_initialized, num_atoms, energy)
    cd(cwd)
    
    if potential.is_initialized == 0
        potential.is_initialized = 1
    end

    if reshape_coords
        coords = vec(coords)
    end

    return energy[begin]
end

function get_energy_and_gradients(potential::Protonated_Water, coords::AbstractArray; reshape_coords::Bool=false)
    if reshape_coords
        coords = reshape(coords, (3, :))
    end
    num_atoms::Int32 = size(coords, 2)

    cwd = String(pwd())
    cd(dirname(potential.full_lib_path))
    
    energy = Float64[0]
    grads = zeros(Float64, length(coords))
    ccall(potential.energy_and_gradient_function, Cvoid, (Ref{Cdouble}, Ref{Int32}, Ref{Int32}, Ref{Cdouble}, Ref{Cdouble}), coords, potential.is_initialized, num_atoms, energy, grads)
    cd(cwd)
    
    if potential.is_initialized == 0
        potential.is_initialized = 1
    end

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


########################
###      NWChem      ###
########################
include("nwchem_input_generator.jl")
include("electronic_structure_parsers.jl")

struct NWChem <: AbstractPotential
    executable_command::Vector{String}
    nwchem_input::NWChemInput
end

NWChem(executable::String, nwchem_input::NWChemInput) = NWChem(string.(split(executable)), nwchem_input)

NWChem(executable::String, basis::Dict{String, String}, theory::String) = NWChem(executable, NWChemInput(basis, theory))
NWChem(executable::String, basis::Dict{String, String}, theory::Vector{String}) = NWChem(executable, NWChemInput(basis, theory))
NWChem(executable::String, basis::String, theory::String) = NWChem(executable, NWChemInput(basis, theory))
NWChem(executable::String, basis::String, theory::Vector{String}) = NWChem(executable, NWChemInput(basis, theory))
NWChem(nwchem::NWChem) = NWChem(nwchem.executable_command, NWChemInput(nwchem.nwchem_input.basis, nwchem.nwchem_input.theory))

function get_energy(nwchem::NWChem, coords::Matrix{T}, atom_labels::Vector{String}, input_file_name::String="input_1.nw", output_directory::String="nwchem", return_dict::Bool=false) where T <: AbstractFloat
    set_task!(nwchem.nwchem_input, "energy")
    used_input_name::String = write_input_file(nwchem.nwchem_input, coords, atom_labels, input_file_name, output_directory)
    output_name = string(splitext(used_input_name)[1], ".out")
    output_string = read(pipeline(`$(nwchem.executable_command) $used_input_name '&'`), String)
    open(output_name, "w") do io
        write(io, output_string)
    end
    # the above blocks until the job is finished which is what we want so we can read in the file on the next line
    nwchem_output = string.(split(output_string, '\n'))
    
    energies = parse_nwchem_energies(nwchem_output)
    if return_dict
        return energies
    elseif length(nwchem.nwchem_input.theory) == 1
        return energies[nwchem.nwchem_input.theory[1]][1]
    else
        return energies
    end
end

function get_energy(nwchem::NWChem, coords::Vector{Matrix{T}}, atom_labels::Vector{Vector{String}}, input_file_name::String="input_1.nw", output_directory::String="nwchem", return_dict::Bool=false) where T <: AbstractFloat
    set_task!(nwchem.nwchem_input, "energy")
    used_input_name::String = write_input_file(nwchem.nwchem_input, coords, atom_labels, input_file_name, output_directory)
    output_name = string(splitext(used_input_name)[1], ".out")
    output_string = read(pipeline(`$(nwchem.executable_command) $used_input_name '&'`), String)
    open(output_name, "w") do io
        write(io, output_string)
    end
    # the above blocks until the job is finished which is what we want so we can read in the file on the next line
    nwchem_output = string.(split(output_string, '\n'))
    
    energies = parse_nwchem_energies(nwchem_output)
    if return_dict
        return energies
    elseif length(nwchem.nwchem_input.theory) == 1
        return energies[nwchem.nwchem_input.theory[1]]
    else
        return energies
    end
end

function get_energy_and_gradients(nwchem::NWChem, coords::Matrix{T}, atom_labels::Vector{String}, input_file_name::String="input_1.nw", output_directory::String="nwchem", return_dict::Bool=false) where T <: AbstractFloat
    set_task!(nwchem.nwchem_input, "gradient")
    used_input_name::String = write_input_file(nwchem.nwchem_input, coords, atom_labels, input_file_name, output_directory)
    output_name = string(splitext(used_input_name)[1], ".out")
    output_string = read(pipeline(`$(nwchem.executable_command) $used_input_name '&'`), String)
    open(output_name, "w") do io
        write(io, output_string)
    end
    # the above blocks until the job is finished which is what we want so we can read in the file on the next line
    nwchem_output = string.(split(output_string, '\n'))
    
    energies  = parse_nwchem_energies(nwchem_output)
    gradients = parse_nwchem_gradients(nwchem_output)
    if return_dict
        return energies, gradients
    elseif length(nwchem.nwchem_input.theory) == 1
        return energies[nwchem.nwchem_input.theory[1]][1], gradients[nwchem.nwchem_input.theory[1]][1]
    else
        return energies, gradients
    end
end

function get_energy_and_gradients(nwchem::NWChem, coords::Vector{Matrix{T}}, atom_labels::Vector{Vector{String}}, return_dict::Bool=false) where T <: AbstractFloat
    set_task!(nwchem.nwchem_input, "gradient")
    used_input_name::String = write_input_file(nwchem.nwchem_input, coords, atom_labels)
    used_input_name = string(pwd(), used_input_name)
    output_name = string(splitext(used_input_name)[1], ".out")
    run(`$(nwchem.executable_command) $used_input_name '>' output_name '&'`) 
    # the above blocks until the job is finished which is what we want so we can read in the file on the next line
    nwchem_output = readlines(output_name)
    
    energies  = parse_nwchem_energies(nwchem_output)
    gradients = parse_nwchem_gradients(nwchem_output)
    if return_dict
        return energies, gradients
    elseif length(nwchem.nwchem_input.theory) == 1
        return energies[nwchem.nwchem_input.theory[1]], gradients[nwchem.nwchem_input.theory[1]]
    else
        return energies, gradients
    end
end

###  NWChem BSSE Correction Methods  ###

function get_bsse_corrected_energy(nwchem::NWChem, coords::Vector{Matrix{T}}, atom_labels::Vector{Vector{String}}, separate_files_for_bsse::Bool=false) where T <: AbstractFloat
    """
    Takes an NWChem struct and writes the appropriate input files to calculate the BSSE-corrected energy of a supermolecule. Each fragment should be contained in the array of matrices and the associated atom labels in atom_labels.
    """
    set_task!(nwchem.nwchem_input, "energy")
    used_input_names = String[]
    if separate_files_for_bsse
        used_input_names = write_bsse_input_files(nwchem.nwchem_input, coords, atom_labels)
    else
        used_input_names = [write_single_bsse_input_file(nwchem.nwchem_input, coords, atom_labels)]
    end
    output_names = [string(splitext(used_input_names[i])[1], ".out") for i in 1:length(used_input_names)]
    
    output_strings = Array{String}(undef, length(output_names))
    @sync for i in 1:length(output_names)
        @async output_strings[i] = read(pipeline(`$(nwchem.executable_command) $(used_input_names[i]) '&'`), String)
    end

    # write out the the results for future reference
    for i in 1:length(output_strings)
        open(output_names[i], "w") do io
            write(io, output_strings[i])
        end
    end
    nwchem_outputs = readlines.(output_names)
    
    all_energies = Float64[]
    if separate_files_for_bsse
        all_energies = [parse_nwchem_energies(nwchem_outputs[i])[nwchem.nwchem_input.theory[1]][1] for i in 1:length(nwchem_outputs)]
    else
        all_energies = parse_nwchem_energies(nwchem_outputs[1])[nwchem.nwchem_input.theory[1]]
    end
    # now combine everything into one bsse-corrected energy
    return @views(all_energies[1] + sum(all_energies[2:(length(coords)+1)] - all_energies[(length(coords)+2):(2*length(coords)+1)]))
end

function get_bsse_corrected_energy_and_gradients(nwchem::NWChem, coords::Vector{Matrix{T}}, atom_labels::Vector{Vector{String}}, separate_files_for_bsse::Bool=false) where T <: AbstractFloat
    """
    Takes an NWChem struct and writes the appropriate input files to calculate the BSSE-corrected energy and gradients of a supermolecule. Each fragment should be contained in the array of matrices and the associated atom labels in atom_labels.
    """
    set_task!(nwchem.nwchem_input, "gradient")

    used_input_names = String[]
    if separate_files_for_bsse
        used_input_names = write_bsse_input_files(nwchem.nwchem_input, coords, atom_labels)
    else
        used_input_names = [write_single_bsse_input_file(nwchem.nwchem_input, coords, atom_labels)]
    end
    output_names = [string(splitext(used_input_names[i])[1], ".out") for i in 1:length(used_input_names)]
    output_strings = Array{String}(undef, length(output_names))
    @sync for i in 1:length(output_names)
        @async output_strings[i] = read(pipeline(`$(nwchem.executable_command) $(used_input_names[i]) '&'`), String)
    end

    # write out the the results for future reference
    for i in 1:length(output_strings)
        open(output_names[i], "w") do io
            write(io, output_strings[i])
        end
    end
    nwchem_outputs = readlines.(output_names)
    
    all_energies = Float64[]
    all_gradients = Matrix{Float64}[]
    if separate_files_for_bsse
        all_energies = [parse_nwchem_energies(nwchem_outputs[i])[nwchem.nwchem_input.theory[1]][1] for i in 1:length(nwchem_outputs)]
        all_gradients = [parse_gradients(nwchem_outputs[i])[nwchem.nwchem_input.theory[1]][1] for i in 1:length(nwchem_outputs)]
    else
        all_energies = parse_nwchem_energies(nwchem_outputs[1])[nwchem.nwchem_input.theory[1]]
        all_gradients = parse_nwchem_gradients(nwchem_outputs[1])[nwchem.nwchem_input.theory[1]]
    end
    # now combine everything into one bsse-corrected energy and gradients
    return @views(all_energies[1] + sum(all_energies[2:(length(coords)+1)] - all_energies[(length(coords)+2):(2*length(coords)+1)])), @views(all_gradients[1] + hcat(all_gradients[2:(length(coords)+1)]...) - sum(all_gradients[(length(coords)+2):(2*length(coords)+1)]))
end

function get_bsse_corrected_gradients!(nwchem::NWChem, grads::Matrix{T}, coords::Vector{Matrix{T}}, atom_labels::Vector{Vector{String}}, separate_files_for_bsse::Bool=false) where T <: AbstractFloat
    grads[:] = get_bsse_corrected_energy_and_gradients(nwchem, coords, atom_labels, separate_files_for_bsse)[2]
end

function get_approximate_bsse_corrected_energy(nwchem::NWChem, coords::Matrix{T}, atom_labels::Vector{String}, approximate_bsse_correction_function::Function) where T <: AbstractFloat
    return get_energy(nwchem, coords, atom_labels) + approximate_bsse_correction_function(coords, atom_labels)
end

function get_approximate_bsse_corrected_gradients!(nwchem::NWChem, grads::Matrix{T}, coords::Matrix{T}, atom_labels::Vector{String}, approximate_bsse_correction_function::Function) where T <: AbstractFloat
    """
    Uses the approximate_bsse_correction_function to correct for BSSE rather than doing the complete BSSE correction via ab initio.
    
    Note that the approximate_bsse_correction_function should return approximate BSSE-corrected gradients for the entire system, not just particular atom pairs.
    """
    grads[:] = get_energy_and_gradients(nwchem, coords, atom_labels)[2] + approximate_bsse_correction_function(coords, atom_labels)
end

############################
########## Q-Chem ##########
############################

struct QChem <: AbstractPotential
    executable_command::Vector{String}
    rem_input::String
    labels::Vector{String}
    charge::Int
    multiplicity::Int
    ofile_name::String
    QChem(executable_command::String, rem_input::String, xyz_file::String, charge::Int, multiplicity::Int, ofile_name::String) = new(split(executable_command), read(rem_input, String), read_xyz(xyz_file)[2][1], charge, multiplicity, ofile_name)
end

function get_energy_and_gradients(qchem::QChem, coords::Matrix{Float64})
    geom_string = geometry_to_string(coords, qchem.labels)
    used_input_name = next_unique_name(qchem.ofile_name)
    open(used_input_name, "w") do io
        write(io, "\$molecule\n")
        write(io, string(qchem.charge, " ", qchem.multiplicity, "\n"))
        write(io, geom_string)
        write(io, "\$end\n\n")
        write(io, qchem.rem_input)
    end
    
    output_name = string(splitext(used_input_name)[1], ".out")
    output_string = read(pipeline(`$(qchem.executable_command) $(used_input_name)`), String)

    # write out the the results for future reference
    open(output_name, "w") do io
        write(io, output_string)
    end
    qchem_output = readlines.(output_name)

    energy = parse_qchem_energies(qchem_output)
    gradients = parse_qchem_gradients(qchem_output)

    return energy["dft"][1], gradients["dft"][1]
end
