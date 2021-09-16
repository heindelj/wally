using Combinatorics, Distributed, UUIDs
include("call_potential.jl")

struct MBEPotential{T} <: AbstractPotential
    potential::T
    order::Int
end

MBEPotential(mbe_potential::MBEPotential) = MBEPotential(typeof(mbe_potential.potential)(mbe_potential.potential), mbe_potential.order)

function get_many_body_geometries(coords::AbstractArray, order::Int)
    """
    Builds an array of 3xN arrays where N is the number of atoms in each geometry
    at each index. Makes the geometries for order N. These are then returned
    for whatever processing should be done on each geometry.

    Coords should return a sub-system when indexed by a single index. That is,
    coords is an array of arrays.
    """
    subsystem_indices = [combinations([1:length(coords)...], order)...]
    subsystem_combos = []
    if eltype(coords) <: AbstractMatrix
        subsystem_combos = Vector{eltype(coords)}(undef, length(subsystem_indices))
        for i in 1:length(subsystem_indices)
            subsystem_combos[i] = hcat(getindex(coords, subsystem_indices[i])...)
        end
    elseif eltype(eltype(coords)) <: AbstractString
        subsystem_combos = Vector{eltype(coords)}(undef, length(subsystem_indices))
        for i in 1:length(subsystem_indices)
            subsystem_combos[i] = vcat(getindex(coords, subsystem_indices[i])...)
        end
    else
        @assert false "Can only combine AbstractMatrix and String types right now."
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

function get_energy(mbe_potential::MBEPotential, coords::AbstractVector{Matrix{T}}, return_mbe_data::Bool=false, copy_construct_potential::Bool=false; kwargs...) where T <: Real
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
            for j in 1:length(all_subsystems[i])
                energies[i] += get_energy(typeof(mbe_potential.potential)(mbe_potential.potential), all_subsystems[i][j]; kwargs...)
            end
        end
    else
        for i in length(all_subsystems):-1:1
            for j in 1:length(all_subsystems[i])
                energies[i] += get_energy(mbe_potential.potential, all_subsystems[i][j]; kwargs...)
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

function get_gradients(mbe_potential::MBEPotential, coords::AbstractVector{Matrix{T}}, return_mbe_data=false, copy_construct_potential=false; kwargs...) where T <: Real
    """
    Forms the n-body geometries and calls the potential belonging to
    mbe_potential on each of them to return the many-body gradients.
    Note that coords is an array of sub-systems (the fragments).
    """
    summed_gradients = [zeros((3, sum(size.(coords, 2)))) for _ in 1:mbe_potential.order]
    all_subsystems = get_many_body_geometries.((coords,), [1:mbe_potential.order...])
    subsystem_energies = [zero.(all_subsystems[i]) for i in 1:length(all_subsystems)]
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

function get_gradients!(potential::MBEPotential, storage::AbstractArray, coords::AbstractVector{Matrix{T}}, copy_construct_potential::Bool=false; kwargs...) where T <: Real
    storage[:] = get_gradients(potential, coords, false, copy_construct_potential; kwargs...)
end

function get_energy_and_gradients(mbe_potential::MBEPotential, coords::AbstractVector{Matrix{T}}, return_mbe_data=false, copy_construct_potential::Bool=false; kwargs...) where T <: Real
    """
    Forms the n-body geometries and calls the potential belonging to
    mbe_potential on each of them to return the many-body energy and gradients.
    Note that coords is an array of sub-systems (the fragments).
    """
    energies = zeros(mbe_potential.order)
    summed_gradients = [zeros((3, sum(size.(coords, 2)))) for _ in 1:mbe_potential.order]
    all_subsystems = get_many_body_geometries.((coords,), [1:mbe_potential.order...])
    subsystem_energies = [[zero(Float64) for _ in 1:length(all_subsystems[i])] for i in 1:length(all_subsystems)]
    subsystem_gradients = [zero.(all_subsystems[i]) for i in 1:length(all_subsystems)]
    # Need to switch to pmap and fetching the futures for parallelization
    if copy_construct_potential
        for i in length(all_subsystems):-1:1
            for j in 1:length(all_subsystems[i])
                subsystem_energies[i][j], subsystem_gradients[i][j] = get_energy_and_gradients(typeof(mbe_potential.potential)(mbe_potential.potential), all_subsystems[i][j]; kwargs...)
            end
        end
    else
        for i in length(all_subsystems):-1:1
            for j in 1:length(all_subsystems[i])
                subsystem_energies[i][j], subsystem_gradients[i][j] = get_energy_and_gradients(mbe_potential.potential, all_subsystems[i][j]; kwargs...)
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
        energies[i] = sum(subsystem_energies[i])
    end

    mbe_gradient_data = get_mbe_data_from_subsystem_sums(summed_gradients, length(coords))
    mbe_energy_data::Array{Float64, 1} = get_mbe_data_from_subsystem_sums(energies, length(coords))
    
    if return_mbe_data
        return mbe_energy_data, mbe_gradient_data
    else
        return sum(mbe_energy_data), sum(mbe_gradient_data)
    end
end

function get_energy_and_gradients(mbe_potential::MBEPotential{NWChem}, coords::AbstractVector{Matrix{T}}, labels::AbstractVector{Vector{String}}, worker_pool::Vector{Int}=[1], return_mbe_data::Bool=false, return_order_n_only::Bool=false) where T <: AbstractFloat
    """
    Forms the n-body geometries and calls the nwchem potential belonging to
    mbe_potential on each of them to return the many-body energy and gradients.
    Note that coords is an array of sub-systems (the fragments).

    if return_order_n_only is true, then only the mbe_potential.order data
    will be returned. i.e. you will get back the 3-body energies and gradients
    if that's the order of MBE requested. This takes precedent over returning
    all MBE data.
    """
    @assert length(coords) == length(labels) "Fragment coordinates and labels are not the same length. The labels should be an array of arrays of all atom labels, as should the geometries."
    energies = zeros(mbe_potential.order)
    summed_gradients = [zeros((3, sum(size.(coords, 2)))) for _ in 1:mbe_potential.order]
    all_subsystems = get_many_body_geometries.((coords,), [1:mbe_potential.order...])
    all_subsystem_labels = get_many_body_geometries.((labels,), [1:mbe_potential.order...])

    subsystem_energies, subsystem_gradients = poll_and_spawn_nwchem_mbe_calculations(mbe_potential.potential, all_subsystems, all_subsystem_labels, mbe_potential.order, length(labels), worker_pool)

    # now put the subsystem forces into the appropriate indices of total forces
    for i in 1:mbe_potential.order
        subsytem_indices = force_indices(i, coords)
        @assert length(subsytem_indices) == length(subsystem_gradients[i]) "Indices and gradients aren't same length."
        for j in 1:length(subsytem_indices)
            @views summed_gradients[i][:, subsytem_indices[j]] += subsystem_gradients[i][j]
        end
        energies[i] = sum(subsystem_energies[i])
    end

    mbe_gradient_data = get_mbe_data_from_subsystem_sums(summed_gradients, length(coords))
    mbe_energy_data   = get_mbe_data_from_subsystem_sums(energies, length(coords))
    
    if return_order_n_only
        return mbe_energy_data[mbe_potential.order], mbe_gradient_data[mbe_potential.order]
    elseif return_mbe_data
        return mbe_energy_data, mbe_gradient_data
    else
        return sum(mbe_energy_data), sum(mbe_gradient_data)
    end
end

function get_energy_and_gradients(potential_dict::Dict{Int, <:AbstractPotential}, coords::AbstractVector{Matrix{T}}, labels::AbstractVector{Vector{String}}, available_workers::Vector{Int}=[1], use_max_order_on_full_system::Bool=true) where T <: AbstractFloat
    """
    This calls the get_energy_and_gradients function for each method in the
    mbe_potential_dict dictionary. We will first determine if some of the
    calculations are redundant. That is, if you request "scf" for 3-body, then
    we will check if 2-body is a post-HF method, in which case the energies
    and gradients for the many-body part of the 3-body term will be taken from
    the earlier calculations.

    The option use_max_order_on_full_system means that if we only specify up
    to 3-body, then we will calculate the entire system with this method and
    calculate the 2-body MBE to obtain the 3- to N-body terms by subtraction.
    """
    all_mbe_orders = sort([keys(potential_dict)...])
    if length(available_workers) < length(all_mbe_orders)
        # return by running the serial version here
    end

    # make all of the main threads and the workers associated with those threads
    # TODO: Make this determine if the mbe order is NWChem (or some other method
    # which parallelizes across workers) and only give the workers to them since
    # all of the classical potentials don't do anything with workers.
    worker_pools  = [Int[] for _ in 1:length(all_mbe_orders)]
    num_workers::Int = length(available_workers)
    for i in 1:num_workers
        push!(worker_pools[(i % length(all_mbe_orders)) + 1], pop!(available_workers))
    end

    mbe_energies  = zeros(length(all_mbe_orders))
    mbe_gradients = [zeros((3, sum(size.(coords, 2)))) for _ in 1:length(all_mbe_orders)]
    if use_max_order_on_full_system
        push!(mbe_energies, 0.0) 
        push!(mbe_gradients, zeros((3, sum(size.(coords, 2))))) 
    end

    @sync begin
        for (i, mbe_order) in enumerate(all_mbe_orders)
            if mbe_order == minimum(all_mbe_orders)
                @async mbe_energies[i], mbe_gradients[i] = typeof(potential_dict[mbe_order]) == NWChem ? get_energy_and_gradients(MBEPotential(potential_dict[mbe_order], mbe_order), coords, labels, worker_pools[i]) : get_energy_and_gradients(MBEPotential(potential_dict[mbe_order], mbe_order), coords, false, true)
            elseif mbe_order < maximum(all_mbe_orders) && mbe_order > minimum(all_mbe_orders)
                @async mbe_energies[i], mbe_gradients[i] = typeof(potential_dict[mbe_order]) == NWChem ? get_energy_and_gradients(MBEPotential(potential_dict[mbe_order], mbe_order), coords, labels, worker_pools[i], false, true) : get_energy_and_gradients(MBEPotential(potential_dict[mbe_order], mbe_order), coords, false, true)
            else
                if use_max_order_on_full_system
                    @async mbe_energies[i], mbe_gradients[i] = typeof(potential_dict[mbe_order]) == NWChem ? get_energy_and_gradients(MBEPotential(potential_dict[mbe_order], mbe_order-1), coords, labels, worker_pools[i]) : get_energy_and_gradients(MBEPotential(potential_dict[mbe_order], mbe_order-1), coords, false, true)
                    mbe_energies[i+1], mbe_gradients[i+1] = typeof(potential_dict[mbe_order]) == NWChem ? fetch(spawn_nwchem_mbe_job(potential_dict[mbe_order], hcat(coords...), append!(String[], labels...), "full_calculation.nw", 2)) : get_energy_and_gradients(potential_dict[mbe_order], hcat(coords...))
                else
                    @async mbe_energies[i], mbe_gradients[i] = typeof(potential_dict[mbe_order]) == NWChem ? get_energy_and_gradients(MBEPotential(potential_dict[mbe_order], mbe_order), coords, labels, worker_pools[i], false, true) : get_energy_and_gradients(MBEPotential(potential_dict[mbe_order], mbe_order), coords, false, true)
                end
            end
        end
    end

    #mbe_energies, mbe_gradients = collect(zip(fetch.(future_energies_and_gradients)...))
    # get the difference from full system calculation with mbe up to max_order-1 and
    # add to the sum of everything up to that point
    if use_max_order_on_full_system
        return sum(@view(mbe_energies[1:end-2])) + mbe_energies[end] - mbe_energies[end-1], sum(@view(mbe_gradients[1:end-2])) + mbe_gradients[end] - mbe_gradients[end-1]
    end
    return sum(mbe_energies), sum(mbe_gradients)
end

# THIS FUNCTION BELOW MIGHT BE A WASTE BECAUSE NWCHEM CURRENTLY DOES NOT DO
# THE ANALYTIC GRADIENTS FOR SCF WHEN MP2 GRADIENTS ARE REQUESTED BECAUSE
# IT DOESN'T HAVE TO. WE SHOULD VERIFY THIS IS ACTUALLY THE CASE THOUGH...
function identify_necessary_mbe_tasks(mbe_potential_dict::Dict{Int, AbstractPotential}, theory_name::String)
    """
    Basically what needs to happen here is whenever something likes CCSD(T)
    or MP2 appears in a lower-order term, but scf appears in a higher-order
    term, I need to make sure we also explicitly calculate the gradients for
    the scf calculations (to force NWChem to print them).
    
    I'm just going to explicitly do every case because there's not going to
    be a clear and elegant way to do this so I'll just be explicit.
    """
    lowest_order = minimum(keys(method_by_order))
    nwchem_potentials::Dict{String, Int} = Dict()
    for (mbe_order, potential) in pairs(mbe_potential_dict)
        if typeof(potential) == NWChem 
            @assert length(potential.theory) == 1 "Please only give one theory per nwchem potential for the MBE. The unique calculations will be determined and having multiple theories makes it ambiguous what calculations we should actually run."
            nwchem_potentials[potential.theory[1]] = mbe_order
        end
    end    
    if "scf" in keys(nwchem_potentials) && nwchem_potentials["scf"] > lowest_order
        if "mp2" in keys(nwchem_potentials) && nwchem_potentials["mp2"] < nwchem_potentials["scf"]
            push!(mbe_potential_dict[nwchem_potentials["mp2"]].nwchem_input.theory, "scf")
        end
        if "ccsd" in keys(nwchem_potentials) && nwchem_potentials["ccsd"] < nwchem_potentials["scf"]
            push!(mbe_potential_dict[nwchem_potentials["ccsd"]].nwchem_input.theory, "scf")
        end
    end
end

function poll_and_spawn_nwchem_mbe_calculations(nwchem::NWChem, all_subsystem_coords::Vector{Vector{Matrix{T}}}, all_subsystem_labels::Vector{Vector{Vector{String}}}, max_mbe_order::Int, num_fragments::Int, worker_pool::Vector{Int}, l=ReentrantLock()) where T <: AbstractFloat
    """
    Polls all tasks spawned at workers to see if they are finished. If so,
    immediately spawn a new calculation for this worker to do and then fetch
    the results from the appropriate future and store the results in the
    appropriate accumulation locations for the final MBE.
    
    There is a small amount of possible latency associated with storing the
    gradients because we have to determine the right indices for each sub-system
    and put the gradients there.
    """
    # TODO: If it turns out that it is hard to time the completion of various sub-system
    # calculations using just the number of resources passed to NWChem, then we should
    # be able to pass a dictionary containing the status of each worker which is set to
    # busy or not busy. So, once a whole pool is finished with the order of the MBE it
    # was originally assigned, we can assign those workers new calculations by putting
    # them in the available workers array and assigning them a calculation to carry out
    # We will need to be careful about the whole pid thing, but I think just pushing the
    # new worker ids into the array will already work...

    # If we only have the main thread, then add workers to handle the tasks
    lock(l)
    try
        if length(worker_pool) == 1
            # adds one worker for the main thread. If you're here you apparently want
            # to run single-threaded. Otherwise, pass a worker_pool to the function.
            # We then include this file at the new worker.
            worker_pool = addprocs(1)
            @everywhere worker_pool include(@__FILE__)
        end
    finally
        unlock(l)
    end

    number_of_calculations::Int = sum(length.(all_subsystem_labels))
    future_results = [Array{Future}(undef, length(all_subsystem_labels[i])) for i in 1:length(all_subsystem_labels)]
    current_mbe_index::Int = 1
    current_fragment_index::Int = 1

    # stores the worker id, mbe index, and fragment index for active job
    active_job_array::Array{Tuple{Int, Int, Int}} = []
    
    number_of_launched_calculations::Int = 0
    # Initialize all of the workers with a calculation (up to number of jobs)
    for i in 1:number_of_calculations
        number_of_launched_calculations += 1
        # spawn the next fragment calculation
        future_results[current_mbe_index][current_fragment_index] = spawn_nwchem_mbe_job(nwchem, all_subsystem_coords[current_mbe_index][current_fragment_index], all_subsystem_labels[current_mbe_index][current_fragment_index], string("input_", current_mbe_index, "_", current_fragment_index, ".nw"), worker_pool[i])
        push!(active_job_array, (worker_pool[i], current_mbe_index, current_fragment_index))

        # increment the mbe index if all fragments at that order have been handled
        # otherwise increment the fragment index
        if current_fragment_index == length(all_subsystem_labels[current_mbe_index])
            current_mbe_index += 1
            current_fragment_index = 1
        else
            current_fragment_index += 1
        end
    end
    # loop over all active jobs until they are all completed
    while number_of_launched_calculations != number_of_calculations
        for (job_index, (pid, mbe_index, fragment_index)) in enumerate(active_job_array)
            if isready(future_results[mbe_index][fragment_index])
                number_of_launched_calculations += 1
                # spawn the new job and store the indices for this job
                future_results[current_mbe_index][current_fragment_index] = spawn_nwchem_mbe_job(nwchem, all_subsystem_coords[current_mbe_index][current_fragment_index], all_subsystem_labels[current_mbe_index][current_fragment_index], string("input_", current_mbe_index, "_", current_fragment_index, ".nw"), pid)
                active_job_array[job_index] = (pid, current_mbe_index, current_fragment_index)
                # increment the mbe index if all fragments at that order have been handled
                # otherwise increment the fragment index
                if current_fragment_index == length(all_subsystem_labels[current_mbe_index])
                    current_mbe_index += 1
                    current_fragment_index = 1
                else
                    current_fragment_index += 1
                end
            end
        end
    end
    # fetch all of the data and store at appropriate locations and process
    # everything into final mbe energies and gradients.
    subsystem_energies = [[zero(Float64) for _ in 1:length(all_subsystem_coords[i])] for i in 1:length(all_subsystem_coords)]
    subsystem_gradients = [zero.(all_subsystem_coords[i]) for i in 1:length(all_subsystem_coords)]
    for i in 1:length(future_results)
        for j in 1:length(future_results[i])
            data = fetch(future_results[i][j])
            subsystem_energies[i][j] = data[1]
            subsystem_gradients[i][j] = data[2]
        end
    end
    return subsystem_energies, subsystem_gradients
end

@inline function spawn_nwchem_mbe_job(nwchem::NWChem, coords::Matrix{T}, labels::Vector{String}, input_file::String, pid::Int) where T <: AbstractFloat
    """
    Spawns an nwchem job at the specified process id and returns a future 
    to the result. 
    Notice that the NWChem struct can be modified so we can control how many 
    processes are used when launching NWChem.
    
    All of the data passed to this function is also passed over to the 
    corresponding process. If this turns out to be a problem producing too much 
    latency, then we may want to consider converting to more of a 
    queueing system where we deposit a whole bunch of tasks at each worker, 
    and then if each worker runs out of jobs, it polls the other workers for 
    available jobs.
    """
    return @spawnat pid get_energy_and_gradients(nwchem, coords, labels, string(splitext(input_file)[1], "_", nwchem.nwchem_input.theory[1], "_", string(values(nwchem.nwchem_input.basis)...), "_at_worker_", pid, "_", string(UUIDs.uuid4()), ".nw"))
end
