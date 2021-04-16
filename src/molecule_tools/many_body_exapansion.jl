using Combinatorics, Distributed, SharedArrays
include("call_potential.jl")

struct MBEPotential <: AbstractPotential
    potential::AbstractPotential
    order::Int
end

function get_many_body_geometries(coords::AbstractArray, order::Int)
    """
    Builds an array of 3xN arrays where N is the number of atoms in each geometry
    at each index. Makes the geometries for order N. These are then returned
    for whatever processing should be done on each geometry.

    Coords should return a sub-system when indexed by a single index. That is,
    coords is an array of arrays.
    """
    subsystem_indices = [combinations([1:length(coords)...], order)...]
    subsystem_combos = Array{Array{Float64, 2}, 1}(undef, length(subsystem_indices))
    for i in 1:length(subsystem_indices)
        subsystem_combos[i] = hcat(getindex(coords, subsystem_indices[i])...)
    end
    return subsystem_combos
end

function get_mbe_data_from_subsystem_sums(data::AbstractArray, num_fragments::Int)
    """
    Takes the sums over some property (probably energies or forces)
    and applies the appropriate weights to each term, returning the final value.
    """
    mbe_data = [zero(data[begin]) for _ in 1:length(data)]
    mbe_data[begin] = data[begin]
    for i_mbe in 1:(length(data)-1)
        for i in 0:i_mbe
            mbe_data[i_mbe+1] += (-1)^i * binomial(num_fragments-(i_mbe+1)+i, i) * data[i_mbe-i+1]
        end
    end
    return mbe_data
end

function force_indices(mbe_order::Int, fragments::AbstractArray)
    """
    https://github.com/heindelj/pyMD/blob/master/py_MD/Fragments.py
    """
    num_fragments::Int = length(fragments)
    fragment_index_array = [1:num_fragments...]
    subsystem_indices = [combinations([fragment_index_array...], mbe_order)...]

    atoms_in_all_preceding_fragments = Array{Int, 1}()
    distance_until_now::Int = 1
    for (i, fragment) in enumerate(fragments)
        if i > 1
            distance_until_now = atoms_in_all_preceding_fragments[i-1]
        end
        push!(atoms_in_all_preceding_fragments, size(fragment, 2) + distance_until_now)
        atoms_in_all_preceding_fragments[i] -= atoms_in_all_preceding_fragments[begin]
    end

    atom_index_array = Array{Array{Int, 1}, 1}()
    for fragment_indices in subsystem_indices
        list_of_lists = [[atoms_in_all_preceding_fragments[i]:(atoms_in_all_preceding_fragments[i] + size(fragments[i], 2) - 1)...] .+ 1 for i in fragment_indices]
        list_of_lists = [item for sublist in list_of_lists for item in sublist]
        push!(atom_index_array, list_of_lists)
    end
    return atom_index_array
end

function get_energy(mbe_potential::MBEPotential, coords::AbstractArray{Array{Float64, 1}}; return_mbe_data=false, copy_construct_potential=false, kwargs...)
    """
    Forms the n-body geometries and calls the potential belonging to
    mbe_potential on each of them to return the many-body energy.
    Note that coords is an array of sub-systems (the fragments).
    """
    energies = zeros(mbe_potential.order)
    all_subsystems = get_many_body_geometries.((coords,), [1:mbe_potential.order...])

    # iterate backwards so the more expensive calculations get queued first.
    if copy_construct_potential
        for i in length(all_subsystems):-1:1
            energies[i] = @distributed (+) for j in 1:length(all_subsystems[i])
                get_energy(typeof(mbe_potential.potential)(mbe_potential.potential), all_subsystems[i][j]; kwargs...)
            end
        end
    else
        for i in length(all_subsystems):-1:1
            energies[i] = @distributed (+) for j in 1:length(all_subsystems[i])
                get_energy(mbe_potential.potential, all_subsystems[i][j]; kwargs...)
            end
        end
    end

    mbe_data::Array{Float64, 1} = get_mbe_data_from_subsystem_sums(energies, length(coords))
    if return_mbe_data
        return mbe_data
    else
        return sum(mbe_data)
    end
end

function get_gradients(mbe_potential::MBEPotential, coords::AbstractArray; return_mbe_data=false, copy_construct_potential=false, kwargs...)
    """
    Forms the n-body geometries and calls the potential belonging to
    mbe_potential on each of them to return the many-body gradients.
    Note that coords is an array of sub-systems (the fragments).
    """
    summed_gradients = [zeros((3, sum(size.(coords, 2)))) for _ in 1:mbe_potential.order]
    all_subsystems = get_many_body_geometries.((coords,), [1:mbe_potential.order...])
    subsystem_gradients = [zero.(all_subsystems[i]) for i in 1:length(all_subsystems)]
    if copy_construct_potential
        for i in length(all_subsystems):-1:1
            @sync @distributed for j in 1:length(all_subsystems[i])
                subsystem_gradients[i][j] = get_gradients(typeof(mbe_potential.potential)(mbe_potential.potential), all_subsystems[i][j]; kwargs...)
            end
        end
    else
        for i in length(all_subsystems):-1:1
            @sync @distributed for j in 1:length(all_subsystems[i])
                subsystem_gradients[i][j] = get_gradients(mbe_potential.potential, all_subsystems[i][j]; kwargs...)
            end
        end
    end

    # now put the subsystem forces into the appropriate indices of total forces
    for i in 1:mbe_potential.order
        subsytem_indices = force_indices(i, coords)
        @assert length(subsytem_indices) == length(subsystem_gradients[i]) "Indices and gradients aren't same length."
        for j in 1:length(subsytem_indices)
            @views summed_gradients[i][:, subsytem_indices[j]] += subsystem_gradients[i][j]
        end
    end

    mbe_data = get_mbe_data_from_subsystem_sums(summed_gradients, length(coords))
    
    if return_mbe_data
        return mbe_data
    else
        return sum(mbe_data)
    end
end

function get_gradients!(potential::MBEPotential, storage::AbstractArray, coords::AbstractArray; copy_construct_potential::Bool=false, kwargs...)
    storage[:] = get_gradients(potential, coords, copy_construct_potential=copy_construct_potential; kwargs...)
end