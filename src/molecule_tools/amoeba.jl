using Combinatorics
include("read_xyz.jl")
include("many_body_expansion.jl")

function single_point_energy(coords::AbstractMatrix{Float64}, labels::AbstractVector{String})
    exe = "/home/heindelj/installs/tinker_exes/analyze"
    params = "/home/heindelj/installs/tinker/params/amoebabio18.prm"
    write_xyz_with_connectivity_and_atom_types("sp_tinker.xyz", labels, coords)
    tinker_output = read(`$exe sp_tinker.xyz $params E`, String)
    for line in split(tinker_output, "\n")
        if occursin("Total Potential Energy", line)
            return parse(Float64, split(line)[5])
        end
    end
    @warn "Never found the total potential energy. Calculation likely failed. Returning 0.0."
    return 0.0
end

function many_single_point_energies(coords::Vector{Matrix{Float64}}, labels::Vector{Vector{String}})
    exe = "/home/heindelj/installs/tinker_exes/analyze"
    params = "/home/heindelj/installs/tinker/params/amoebabio18.prm"
    
    # Tinker can't deal with changes in the number of atoms if all xyz are in the same file.
    # So, I scan through and split up all of the files so that they have identical labels in
    # each file. We keep track of the index each geometry had originally so that we can
    # reconstruct the energies as if we had evaluated them all at once.

    index_map = Dict{Vector{String}, Vector{Int}}()

    for i in eachindex(labels)
        if haskey(index_map, labels[i])
            push!(index_map[labels[i]], i)
        else
            index_map[labels[i]] = Int[i]
        end
    end

    for (i_key, key) in enumerate(keys(index_map))
        coords_by_key = coords[index_map[key]]
        labels_by_key = labels[index_map[key]]
        for i in eachindex(coords_by_key)
            write_xyz_with_connectivity_and_atom_types(string("many_sp_tinker_", i_key, ".xyz"), labels_by_key[i], coords_by_key[i], i == 1 ? "w" : "a")
        end
    end
    tinker_outputs = Dict{Vector{String}, String}()
    for (i_key, key) in enumerate(keys(index_map))
        command = [exe, string("many_sp_tinker_", i_key, ".xyz"), params, "E"]
        tinker_outputs[key] = read(Cmd(command), String)
    end

    all_energies = zeros(length(coords))
    for key in keys(tinker_outputs)
        temp_energies = Float64[]
        tinker_output = tinker_outputs[key]
        for line in split(tinker_output, "\n")
            if occursin("Total Potential Energy", line)
                push!(temp_energies, parse(Float64, split(line)[5]))
            end
        end
        all_energies[index_map[key]] = temp_energies
    end
    n_geoms = length(coords)
    n_energies = length(all_energies)
    @assert length(all_energies) == length(coords) "Got $n_geoms but only $n_energies. Something went wrong."

    return all_energies
end

function mbe(coords::AbstractMatrix{Float64}, labels::AbstractVector{String}, fragment_indices::Vector{Vector{Int}}, max_order::Int)
    all_energies = [Float64[] for _ in 1:max_order]
    for i_order in 1:max_order
        all_subsystem_indices = combinations(fragment_indices, i_order)
        energies = zeros(length(all_subsystem_indices))
        for (i, subsystem_indices) in enumerate(all_subsystem_indices)
            flat_indices = reduce(vcat, subsystem_indices)
            energies[i] = single_point_energy(coords[:, flat_indices], labels[flat_indices])
        end
        all_energies[i_order] = energies
    end
    return get_mbe_data_from_subsystem_sums(sum.(all_energies), length(fragment_indices))
end

"""
Compute many-body expansion for many subsystems with the same fragmentation by writing all subsystems
to a single xyz file and calling tinker once. We then recombine the appropriate energies and return
the MBEs for all systems at once.
"""
function many_mbe(coords::Vector{Matrix{Float64}}, labels::Vector{Vector{String}}, fragment_indices::Vector{Vector{Int}}, max_order::Int)
    num_subsystems_per_geom = sum([length(combinations(fragment_indices, i_order)) for i_order in 1:max_order])
    mbe_indices = [[(num_subsystems_per_geom * (i-1) + 1):(num_subsystems_per_geom * i)...] for i in eachindex(coords)]
    all_mbe_geoms = Matrix{Float64}[]
    all_mbe_labels = Vector{String}[]
    for i_geom in eachindex(coords)
        for i_order in 1:max_order
            all_subsystem_indices = combinations(fragment_indices, i_order)
            for subsystem_indices in all_subsystem_indices
                flat_indices = reduce(vcat, subsystem_indices)
                push!(all_mbe_geoms, coords[i_geom][:, flat_indices])
                push!(all_mbe_labels, labels[i_geom][flat_indices])
            end
        end
    end

    # Some more indexing nonsense to get which of the energies belongs at each order.
    # That is, there are seven subsystems for three fragments and max_order=3. We
    # need to be able to figure out which of these energies are from monomers, dimers,
    # and trimers. In general, we have to calculate this.
    nmer_count = [binomial(length(fragment_indices), i_order) for i_order in 1:max_order]
    nmer_index_pattern = Vector{Int}[]
    push!(nmer_index_pattern, [1:nmer_count[1]...])
    for i in 2:length(nmer_count)
        push!(nmer_index_pattern, [(nmer_index_pattern[end][end]+1):(nmer_index_pattern[end][end]+nmer_count[i])...])
    end

    all_sp_energies = many_single_point_energies(all_mbe_geoms, all_mbe_labels)
    all_mbe_subsystem_energies_flat = [[all_sp_energies[mbe_indices[i][i_mbe]] for i_mbe in eachindex(mbe_indices[i])] for i in eachindex(mbe_indices)]
    
    all_mbe_subsystem_energies = Vector{Vector{Float64}}[]
    for i in eachindex(all_mbe_subsystem_energies_flat)
        subsystem_energies = Vector{Float64}[]
        for index_pattern in nmer_index_pattern
            push!(subsystem_energies, all_mbe_subsystem_energies_flat[i][index_pattern])
        end
        push!(all_mbe_subsystem_energies, subsystem_energies)
    end
    all_mbe_energies = [get_mbe_data_from_subsystem_sums(sum.(all_mbe_subsystem_energies[i]), length(fragment_indices)) for i in eachindex(all_mbe_subsystem_energies)]
    return all_mbe_energies
end