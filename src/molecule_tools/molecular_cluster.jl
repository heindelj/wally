using StaticArrays, NearestNeighbors, ProgressBars, ProgressMeter, DelimitedFiles
using StatsBase: countmap
include("covalent_radii.jl")
include("molecular_axes.jl")
include("read_xyz.jl")
include("water_tools.jl")
include("nwchem_input_generator.jl")

"""
The point of this it to somehow represent disjoint molecular
systems. This would include clusters, any aqueous system, etc.

Very often, one wants to do some kind of search based on molecular units
rather than atomic units. So, we construct the molecular sub-systems
based on bonding and then store the position of those molecules as a
point in space. The position of a molecule may be chosen by
different metrics, but by default I just use the centroid of the
cartesian coordinates. Other definitions are almost certainly better
depending on the context.

Bonding is defined on the basis of simple covalent radius criteria.

This allows, for instance, finding the nearest neighbors based on
molecular units rather than the nearest atoms.
"""

struct Cluster
    centers::Vector{SVector{3,Float64}} # location of each molecule
    indices::Vector{Vector{Int}} # maps from molecule indices to full system indices
    geom::Matrix{Float64} # full system coordinates
    labels::Vector{String} # full system labels
end

function is_bonded(distance::Float64, label1::String, label2::String)
    exclusion_list = ["Li", "Na", "K", "Rb", "Cs", "F", "Cl", "Br", "I"]
    if label1 ∉ exclusion_list && label2 ∉ exclusion_list
        return distance < (covalent_radius(label1) + covalent_radius(label2) + 0.4)
    end
    return false
end

"""
neighboring_indices and neighbor_dists are vectors of indices
and distances to that index for all atoms. The indices are sorted
against the distance but we don't utilize that fact.

We use the above arrays to find all bonded neighbors of a particular
atom and then recursively follow the neighbors until the only neighbor
of each atom is itself or an atom contained in the list already.
"""
function find_bonded_atoms!(
    neighboring_indices::Vector{Vector{Int}},
    neighbor_dists::Vector{Vector{Float64}},
    molecular_indices::Vector{Int},
    starting_index::Int,
    labels::Vector{String}
)
    for i in 1:length(neighbor_dists[starting_index])
        neighbor_idx = neighboring_indices[starting_index][i]
        if is_bonded(neighbor_dists[starting_index][i],
            labels[starting_index],
            labels[neighbor_idx])
            if neighbor_idx ∉ molecular_indices
                push!(molecular_indices, neighbor_idx)
                if (neighbor_idx != starting_index)
                    find_bonded_atoms!(neighboring_indices,
                        neighbor_dists,
                        molecular_indices,
                        neighbor_idx,
                        labels)
                end
            end
        end
    end
end

function build_cluster(geom::AbstractMatrix{Float64}, labels::AbstractVector{String}, max_number_of_atoms::Int=10)
    """
    Default method for building a molecular cluster. Uses covalent radius definition of bonding and centroid as position of molecular subunits.
    """
    nl = KDTree(geom)
    indices, dists = knn(nl, geom, length(labels) > max_number_of_atoms + 1 ? max_number_of_atoms + 1 : length(labels), true)

    cluster_indices = Vector{Vector{Int}}()
    for i in 1:length(labels)
        if length(cluster_indices) == 0
            molecular_indices = Int[]
            find_bonded_atoms!(indices, dists, molecular_indices, i, labels)
            push!(cluster_indices, molecular_indices)
        elseif sum(i .∈ cluster_indices) == 0
            molecular_indices = Int[]
            find_bonded_atoms!(indices, dists, molecular_indices, i, labels)
            if isempty(molecular_indices)
                # we should only get here if the atom is in the exclusion list
                # So, we just add that index by itself. 
                push!(molecular_indices, i)
            end
            push!(cluster_indices, molecular_indices)
        end
    end
    cluster_centers = [SVector{3,Float64}(centroid(geom[:, index_set])) for index_set in cluster_indices]
    return Cluster(cluster_centers, cluster_indices, geom, labels)
end

"""
Finds the n nearest neighbors for a collection of indices.
The center_indices are the indices for the cluster, not the geometry
from which the cluster is derived. So, the easiest way to get these
indices is by looking up based on the molecular formula.
"""
function find_n_nearest_neighbors(cluster::Cluster, center_index::Int, n::Int, sortres::Bool=true)
    nl = KDTree(cluster.centers)
    neighbor_indices, _ = knn(nl, cluster.centers[center_index], n + 1, sortres) # the self of something is counted as a neighbor so add one to number requested
    labels_out = Vector{String}[]
    labels_env = Vector{String}[]
    geoms_out = Vector{Float64}[]
    geoms_env = Vector{Float64}[]
    env_indices = setdiff([1:length(cluster.centers)...], neighbor_indices)
    for i in eachindex(neighbor_indices)
        total_indices = reduce(vcat, cluster.indices[neighbor_indices[i]])
        push!(labels_out, cluster.labels[total_indices])
        for vec in eachcol(cluster.geom[:, total_indices])
            push!(geoms_out, vec[:])
        end
    end
    if !isempty(env_indices)
        for i in eachindex(env_indices)
            total_indices = reduce(vcat, cluster.indices[env_indices[i]])
            push!(labels_env, cluster.labels[total_indices])
            for vec in eachcol(cluster.geom[:, total_indices])
                push!(geoms_env, vec[:])
            end
        end
        return reduce(vcat, labels_out), reduce(hcat, geoms_out), reduce(vcat, labels_env), reduce(hcat, geoms_env)
    end
    return reduce(vcat, labels_out), reduce(hcat, geoms_out), String[], Array{Float64}(undef, 3, 0)
end

"""
Finds the neighbors within a certain range for an index.
The center_index is the index for the cluster, not the geometry
from which the cluster is derived. So, the easiest way to get the
index is by looking up based on the molecular formula.
Range is a distance in the same units as that of the cluster.
Excluded indices will exclude the centers in the provided array.
This is useful if you want to find the neighbors in a sub-cluster
you already have without including the neighbors in that sub-cluster.
"""
function find_neighbors_within_range(cluster::Cluster, center_index::Int, range::Float64, sortres::Bool=true)
    nl = KDTree(cluster.centers)
    return inrange(nl, cluster.centers[center_index], range, sortres)
end

function molecules_by_formula(cluster::Cluster, chemical_formula::Vector{String})
    indices_out = Vector{Int}()
    for (i, molecule_indices) in enumerate(cluster.indices)
        if countmap(chemical_formula) == countmap(cluster.labels[molecule_indices])
            push!(indices_out, i)
        end
    end
    return indices_out
end

"""
Finds n nearest neighbors to a molecule by formula. If multiple fragments with that formula are found,
then the one specified by center_index is chosen. Otherwise, just uses the first one.
Returns both the cluster geometry/labels and the environment geometry/labels.
"""
function find_n_nearest_neighbors(cluster::Cluster, chemical_formula::Vector{String}, n::Int, center_index::Int=1, sortres::Bool=true)
    molecule_indices = molecules_by_formula(cluster, chemical_formula)
    if n >= length(cluster.centers)
        n = length(cluster.centers)-1
    end
    return find_n_nearest_neighbors(cluster, molecule_indices[center_index], n, sortres)
end

function write_n_nearest_neighbors(infile::String, chemical_formula::Vector{String}, n::Int, file_name::String="subclusters.xyz")
    _, labels, geoms = read_xyz(infile)
    write_n_nearest_neighbors(geoms, labels, chemical_formula, n, file_name)
end

function write_n_nearest_neighbors(geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, chemical_formula::Vector{String}, n::Int, file_name::String="subclusters.xyz")
    labels_out = [Vector{String}[] for _ in 1:Threads.nthreads()]
    geoms_out = [Matrix{Float64}[] for _ in 1:Threads.nthreads()]

    Threads.@threads for i in ProgressBar(1:length(geoms))
        id = Threads.threadid()
        cluster = build_cluster(geoms[i], labels[i])
        # get all indices corresponding to this chemical formula
        center_indices = molecules_by_formula(cluster, chemical_formula)
        for i_center in center_indices
            labels_frame, geoms_frame = find_n_nearest_neighbors(cluster, i_center, n)
            push!(labels_out[id], labels_frame)
            push!(geoms_out[id], geoms_frame)
        end
    end
    final_labels_out = Vector{String}[]
    final_geoms_out = Matrix{Float64}[]
    append!.((final_labels_out,), labels_out)
    append!.((final_geoms_out,), geoms_out)
    write_xyz(file_name, final_labels_out, final_geoms_out)
end

function get_fragmented_geoms_and_labels(geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}})
    fragmented_geoms = Vector{Matrix{Float64}}[]
    fragment_labels = Vector{Vector{String}}[]
    for i in eachindex(geoms)
        cluster = build_cluster(geoms[i], labels[i])
        fragments = Matrix{Float64}[]
        sub_fragment_labels = Vector{String}[]
        for i_cluster in eachindex(cluster.indices)
            push!(fragments, geoms[i][:, sort(cluster.indices[i_cluster])])
            push!(sub_fragment_labels, labels[i][sort(cluster.indices[i_cluster])])
        end
        push!(fragmented_geoms, fragments)
        push!(fragment_labels, sub_fragment_labels)
    end
    return fragment_labels, fragmented_geoms
end

"""
Samples subclusters from a trajectory centered on the provided
chemical formula. All fragments within the given range are included
in the cluster. The sample will be expanded when molecules in
expand_sample_if_fragment_found are encountered.
Sample metadata returned is a tuple of the frame number from which
a cluster was sampled and the cluster number within that frame.
"""
function sample_random_clusters_within_range(
    geoms::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    chemical_formula::Vector{String},
    radius::Float64,
    expand_sample_if_fragment_found::Vector{Vector{String}},
    expansion_radius::Float64,
    number_of_clusters_to_sample::Int,
    skip_first_n_frames::Int=0,
    resample_if_found_less_than_n_neighbors::Int=5
)
    num_frames = length(geoms)
    @assert skip_first_n_frames < num_frames "You requested we skip $skip_first_n_frames but there are only $num_frames frames."

    fragment_charges = Dict(
        ["O", "H"] => -1,
        ["Cl"]     => -1,
        ["I"]      => -1,
        ["Na"]     =>  1,
    )

    all_sampled_geoms = Matrix{Float64}[]
    all_sampled_labels = Vector{String}[]
    all_environment_geoms = Matrix{Float64}[]
    all_environment_labels = Vector{String}[]
    all_sample_metadata = Tuple{Int, Int, Int}[]

    frame_indices = rand((skip_first_n_frames+1):num_frames, number_of_clusters_to_sample)
    lk = ReentrantLock()
    # Notice that I am using goto statements here.
    # I think it actually accomplishes the task pretty well
    # and it is clear what it is happening. I feel no shame.
    Threads.@threads for i_temp in ProgressBar(eachindex(frame_indices))
        cluster_charge = fragment_charges[chemical_formula]
        i_frame = frame_indices[i_temp]
        cluster = build_cluster(geoms[i_frame], labels[i_frame])
        center_indices = molecules_by_formula(cluster, chemical_formula)
        @label original_sample
        cluster_sample = rand(1:length(center_indices), 1)[1] # This returns a vector, so just get the Int
        neighbor_indices = find_neighbors_within_range(cluster, center_indices[cluster_sample], radius, false)
        if length(neighbor_indices) < resample_if_found_less_than_n_neighbors
            # we only get here if the fragment of interest
            # has a small number of neighbors. This probably
            # indicates the molecule evaporated away or something
            # like that. Just throw this away and resample.
            
            # TODO: Fix the fact this could technically loop infinitely
            # by keeping track of the number of resets and quitting at a
            # certain number.
            @goto original_sample
        end
        fragmented_labels = [cluster.labels[cluster.indices[i]] for i in neighbor_indices]
        fragmented_geom = [cluster.geom[:, cluster.indices[i]] for i in neighbor_indices]
        
        # Now recursively expand the cluster around any fragments
        # from expand_sample_if_fragment_found
        already_expanded = Int[] # stores expansion indices we already dealt with
        @label recursively_expand
        for i in eachindex(expand_sample_if_fragment_found)
            maybe_label_index = findall(x -> x == expand_sample_if_fragment_found[i], fragmented_labels)
            label_index = 0

            # find the appropriate index to expand around
            # but ensure we aren't expanding around the same index
            # we found in the first place. i.e. the one we expanded
            # around on previous iterations of the recusion.
            if !isempty(maybe_label_index)
                for i_found in eachindex(maybe_label_index)
                    if neighbor_indices[maybe_label_index[i_found]] ∉ already_expanded
                        haskey(fragment_charges, fragmented_labels[i_found]) ? cluster_charge += fragment_charges[fragmented_labels[i_found]] : nothing
                        label_index = maybe_label_index[i_found]
                    end
                end
            end
            if label_index == 0
                continue
            end

            expansion_center_index = neighbor_indices[label_index]
            push!(already_expanded, expansion_center_index)
            expanded_indices = find_neighbors_within_range(cluster, expansion_center_index, expansion_radius)
            expanded_indices = setdiff(expanded_indices, neighbor_indices)

            expanded_labels = [cluster.labels[cluster.indices[i]] for i in setdiff(expanded_indices, neighbor_indices)]
            expanded_geom = [cluster.geom[:, cluster.indices[i]] for i in setdiff(expanded_indices, neighbor_indices)]
            fragmented_labels = [fragmented_labels..., expanded_labels...]
            fragmented_geom = [fragmented_geom..., expanded_geom...]
            append!(neighbor_indices, expanded_indices)

            @goto recursively_expand
        end

        environment_labels = [cluster.labels[cluster.indices[i]] for i in setdiff(1:length(cluster.indices), neighbor_indices)]
        environment_geom = [cluster.geom[:, cluster.indices[i]] for i in setdiff(1:length(cluster.indices), neighbor_indices)]
        lock(lk) do
            push!(all_sampled_labels, reduce(vcat, fragmented_labels))
            push!(all_sampled_geoms, reduce(hcat, fragmented_geom))
            push!(all_environment_labels, reduce(vcat, environment_labels))
            push!(all_environment_geoms, reduce(hcat, environment_geom))
            push!(all_sample_metadata, (i_frame, cluster_sample, cluster_charge))
            
            @assert length(all_sampled_labels[end]) + length(all_environment_labels[end]) == length(labels[i_frame])
        end
    end

    return all_sample_metadata, all_sampled_labels, all_sampled_geoms, all_environment_labels, all_environment_geoms
end

function sample_random_clusters_with_n_neighbors(
    geoms::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    chemical_formula::Vector{String},
    num_neighbors::Int,
    number_of_clusters_to_sample::Int,
    skip_first_n_frames::Int=0
)
    num_frames = length(geoms)
    @assert skip_first_n_frames < num_frames "You requested we skip $skip_first_n_frames but there are only $num_frames frames."

    fragment_charges = Dict(
        ["O", "H", "H"] => 0,
        ["O", "H"] => -1,
        ["Cl"]     => -1,
        ["I"]      => -1,
        ["Na"]     =>  1,
    )

    all_sampled_geoms = Matrix{Float64}[]
    all_sampled_labels = Vector{String}[]
    all_environment_geoms = Matrix{Float64}[]
    all_environment_labels = Vector{String}[]
    all_sample_metadata = Tuple{Int, Int, Int}[]

    frame_indices = rand((skip_first_n_frames+1):num_frames, number_of_clusters_to_sample)

    # This is a total hack and dumb. There is a bug where find_n_nearest_neighbors fails for some
    # input but I don't know why. So, I am just assuming the input where it fails is somehow
    # flawed and I generate extra indices which I use as fall back indices to look at when
    # the original ones fail.
    extra_indices = rand((skip_first_n_frames+1):num_frames, number_of_clusters_to_sample)
    cluster_charge = fragment_charges[chemical_formula]
    lk = ReentrantLock()
    Threads.@threads for i_temp in ProgressBar(eachindex(frame_indices))
        i_frame = frame_indices[i_temp]
        @label start
        cluster = build_cluster(geoms[i_frame], labels[i_frame])
        center_indices = molecules_by_formula(cluster, chemical_formula)
        cluster_sample = rand(1:length(center_indices), 1)[1] # This returns a vector, so just get the Int
        try
            cluster_labels, cluster_geoms, env_labels, env_geoms = find_n_nearest_neighbors(cluster, cluster_sample, num_neighbors)
        catch
            i_frame = pop!(extra_indices)
            @goto start
        end
        lock(lk) do
            push!(all_sampled_labels, cluster_labels)
            push!(all_sampled_geoms, cluster_geoms)
            push!(all_environment_labels, env_labels)
            push!(all_environment_geoms, env_geoms)
            push!(all_sample_metadata, (i_frame, cluster_sample, cluster_charge))
            
            @assert length(all_sampled_labels[end]) + length(all_environment_labels[end]) == length(labels[i_frame])
        end
    end
    return all_sample_metadata, all_sampled_labels, all_sampled_geoms, all_environment_labels, all_environment_geoms
end

function locate_shells_near_cluster(cluster_coords::Matrix{Float64}, shell_coords::Matrix{Float64}, cluster_charge::Int)
    full_coords = hcat(cluster_coords, shell_coords)
    num_cluster_atoms = size(cluster_coords, 2)
    nl = KDTree(full_coords)
    shell_indices_to_exclude = Int[]
    num_iterations = 0
    min_distance = 1.0
    while length(shell_indices_to_exclude) != num_cluster_atoms - cluster_charge
        indices_to_exclude = Int[]
        for i in 1:num_cluster_atoms
            @views append!(indices_to_exclude, inrange(nl, cluster_coords[:, i], min_distance + num_iterations * 0.05, false))
        end

        shell_indices_to_exclude   = unique(indices_to_exclude[findall(>(size(cluster_coords, 2)), indices_to_exclude)])
        
        num_iterations += 1
        if num_iterations >= 10
            break
        end
    end
    #if length(shell_indices_to_exclude) != num_cluster_atoms - cluster_charge
    #    write_xyz("failed_test.xyz", ["H" for _ in eachindex(shell_indices_to_exclude)], shell_coords[:, (shell_indices_to_exclude .- size(cluster_coords, 2))])
    #end
    #display(size(shell_coords, 2))
    # minus because a negative charge means an extra shell
    #@assert length(shell_indices_to_exclude) == num_cluster_atoms - cluster_charge string("Expected to find ", num_cluster_atoms - cluster_charge, " shells but found ", length(shell_indices_to_exclude), ". Charge: ", cluster_charge, " N: ", num_cluster_atoms)
    #have to shift all the indices since we found them based on the combined coordinates.
    return shell_indices_to_exclude .- size(cluster_coords, 2)
end

"""
This is a wrapper around sample_random_clusters_within_range
that writes out the sampled clusters and environments
every 100 samples.
"""
function write_random_samples_within_range(
    outfile_prefix::String,
    geoms::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    shell_coords::AbstractVector{Matrix{Float64}},
    chemical_formula::Vector{String},
    radius::Float64,
    expand_sample_if_fragment_found::Vector{Vector{String}},
    expansion_radius::Float64,
    number_of_clusters_to_sample::Int,
    skip_first_n_frames::Int=0
)
    extra_samples = number_of_clusters_to_sample - (number_of_clusters_to_sample ÷ 100) * 100
    # start from zero so we always enter the loop once
    for i in 0:(number_of_clusters_to_sample ÷ 100)
        num_samples = 100
        if i == (number_of_clusters_to_sample ÷ 100)
            num_samples = extra_samples
            if extra_samples == 0
                break
            end
        end
        output = sample_random_clusters_within_range(
            geoms,
            labels,
            chemical_formula,
            radius,
            expand_sample_if_fragment_found,
            expansion_radius,
            num_samples,
            skip_first_n_frames
        )

        sampled_metadata, sampled_labels, sampled_geoms, environment_labels, environment_geoms = output

        # sort the sampled cluster so that hydroxide is at the top #
        for i in eachindex(sampled_geoms)
            sorted_labels, sorted_coords = sort_water_cluster(sampled_geoms[i], sampled_labels[i])
            sampled_labels[i] = sorted_labels
            sampled_geoms[i] = sorted_coords
        end

        write_xyz(
            string(outfile_prefix, ".xyz"),
            [string(length(sampled_labels[i]), "\n", "Frame: ", sampled_metadata[i][1], " Center: ", sampled_metadata[i][2]) for i in eachindex(sampled_labels)],
            sampled_labels,
            sampled_geoms,
            append=true
        )

        mkpath("env_charges")
        for i_env in eachindex(environment_labels)
            core_position_and_charge_matrix = vcat(environment_geoms[i_env], ones((1, length(environment_labels[i_env]))))
            
            shell_indices_to_exclude = locate_shells_near_cluster(sampled_geoms[i_env], shell_coords[sampled_metadata[i_env][1]], sampled_metadata[i_env][3])
            shell_indices = setdiff(1:size(shell_coords[sampled_metadata[i_env][1]], 2), shell_indices_to_exclude)
            shell_positions = shell_coords[sampled_metadata[i_env][1]][:, shell_indices]
            shell_position_and_charge_matrix = vcat(shell_positions, -ones((1, length(shell_indices))))
        
            writedlm(string("env_charges/charges_sample_", i * 100 + i_env, ".xyz"), hcat(core_position_and_charge_matrix, shell_position_and_charge_matrix)')
        end

        # sort the sampled environment so that hydroxide is at the top #
        for i in eachindex(environment_geoms)
            sorted_labels, sorted_coords = sort_water_cluster(environment_geoms[i], environment_labels[i])
            environment_labels[i] = sorted_labels
            environment_geoms[i] = sorted_coords
        end

        write_xyz(
            string(outfile_prefix, "_environment.xyz"),
            [string(length(environment_labels[i]), "\n", "Frame: ", sampled_metadata[i][1], " Center: ", sampled_metadata[i][2]) for i in eachindex(environment_labels)],
            environment_labels,
            environment_geoms,
            append=true
        )
    end
end

function write_random_samples_within_range(
    outfile_prefix::String,
    geoms::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    chemical_formula::Vector{String},
    radius::Float64,
    expand_sample_if_fragment_found::Vector{Vector{String}},
    expansion_radius::Float64,
    number_of_clusters_to_sample::Int,
    skip_first_n_frames::Int=0
)
    extra_samples = number_of_clusters_to_sample - (number_of_clusters_to_sample ÷ 100) * 100
    # start from zero so we always enter the loop once
    for i in 0:(number_of_clusters_to_sample ÷ 100)
        num_samples = 100
        if i == (number_of_clusters_to_sample ÷ 100)
            num_samples = extra_samples
            if extra_samples == 0
                break
            end
        end
        output = sample_random_clusters_within_range(
            geoms,
            labels,
            chemical_formula,
            radius,
            expand_sample_if_fragment_found,
            expansion_radius,
            num_samples,
            skip_first_n_frames
        )

        sampled_metadata, sampled_labels, sampled_geoms, environment_labels, environment_geoms = output

        # sort the sampled cluster so that hydroxide is at the top #
        for i in eachindex(sampled_geoms)
            sorted_labels, sorted_coords = sort_water_cluster(sampled_geoms[i], sampled_labels[i])
            sampled_labels[i] = sorted_labels
            sampled_geoms[i] = sorted_coords
        end

        write_xyz(
            string(outfile_prefix, ".xyz"),
            [string(length(sampled_labels[i]), "\n", "Frame: ", sampled_metadata[i][1], " Center: ", sampled_metadata[i][2]) for i in eachindex(sampled_labels)],
            sampled_labels,
            sampled_geoms,
            append=true
        )

        # sort the sampled environment so that hydroxide is at the top #
        for i in eachindex(environment_geoms)
            sorted_labels, sorted_coords = sort_water_cluster(environment_geoms[i], environment_labels[i])
            environment_labels[i] = sorted_labels
            environment_geoms[i] = sorted_coords
        end

        write_xyz(
            string(outfile_prefix, "_environment.xyz"),
            [string(length(environment_labels[i]), "\n", "Frame: ", sampled_metadata[i][1], " Center: ", sampled_metadata[i][2]) for i in eachindex(environment_labels)],
            environment_labels,
            environment_geoms,
            append=true
        )
    end
end

function write_random_samples_with_n_neighbors(
    outfile_prefix::String,
    geoms::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    chemical_formula::Vector{String},
    num_neighbors::Int,
    number_of_clusters_to_sample::Int,
    skip_first_n_frames::Int=0
)
    extra_samples = number_of_clusters_to_sample - (number_of_clusters_to_sample ÷ 100) * 100
    # start from zero so we always enter the loop once
    for i in 0:(number_of_clusters_to_sample ÷ 100)
        num_samples = 100
        if i == (number_of_clusters_to_sample ÷ 100)
            num_samples = extra_samples
            if extra_samples == 0
                break
            end
        end
        output = sample_random_clusters_with_n_neighbors(
            geoms,
            labels,
            chemical_formula,
            num_neighbors,
            num_samples,
            skip_first_n_frames
        )

        sampled_metadata, sampled_labels, sampled_geoms, environment_labels, environment_geoms = output

        # sort the sampled cluster so that hydroxide is at the top #
        for i in eachindex(sampled_geoms)
            sorted_labels, sorted_coords = sort_water_cluster(sampled_geoms[i], sampled_labels[i])
            sampled_labels[i] = sorted_labels
            sampled_geoms[i] = sorted_coords
        end

        write_xyz(
            string(outfile_prefix, ".xyz"),
            [string(length(sampled_labels[i]), "\n", "Frame: ", sampled_metadata[i][1], " Center: ", sampled_metadata[i][2]) for i in eachindex(sampled_labels)],
            sampled_labels,
            sampled_geoms,
            append=true
        )

        # sort the sampled environment so that hydroxide is at the top #
        for i in eachindex(environment_geoms)
            sorted_labels, sorted_coords = sort_water_cluster(environment_geoms[i], environment_labels[i])
            environment_labels[i] = sorted_labels
            environment_geoms[i] = sorted_coords
        end

        write_xyz(
            string(outfile_prefix, "_environment.xyz"),
            [string(length(environment_labels[i]), "\n", "Frame: ", sampled_metadata[i][1], " Center: ", sampled_metadata[i][2]) for i in eachindex(environment_labels)],
            environment_labels,
            environment_geoms,
            append=true
        )
    end
end

function write_input_files_for_vie_calculations(
    infile_prefix::String,
    cluster_geom_file::String
)
    atom_charges = Dict(
        "O"   => -2,
        "Cl"  => -1,
        "H"   =>  1,
        "Na"  =>  1,
    )

    _, cluster_labels, cluster_geoms = read_xyz(cluster_geom_file)
    
    rem_input_string = "\$rem
jobtype                 sp
method                  wB97M-V
unrestricted            1
basis                   aug-cc-pVDZ
xc_grid        2
scf_max_cycles          500
scf_convergence         6
thresh                  14
s2thresh 14
symmetry                0
sym_ignore              1
mem_total 256000
mem_static 16000
\$end"

    mkpath("qchem_input_files")
    @showprogress for i in eachindex(cluster_labels)
        cluster_charge = sum([atom_charges[label] for label in cluster_labels[i]])

        geom_string = geometry_to_string(cluster_geoms[i], cluster_labels[i])
        open(string("qchem_input_files/", infile_prefix, "_sample_", i, ".in"), "w") do io
            write(io, "\$molecule\n")
            write(io, string(cluster_charge, " ", 1, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string)
            write(io, "\n\$external_charges\n")
            writedlm(io, readdlm(string("env_charges/charges_sample_", i, ".xyz")))
            write(io, "\$end\n\n@@@\n\n")
            write(io, "\$molecule\n")
            write(io, string(cluster_charge+1, " ", 2, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string)
            write(io, "\n\$external_charges\n")
            writedlm(io, readdlm(string("env_charges/charges_sample_", i, ".xyz")))
            write(io, "\$end\n\n")
        end
    end
end

"""
This function assumes that you have generated cluster_samples and the
charges which are used as the environment.
"""
function write_mixed_basis_input_files_for_vie_calculations(
    infile_prefix::String,
    cluster_geom_file::String
)
    atom_charges = Dict(
        "O"   => -2,
        "Cl"  => -1,
        "H"   =>  1,
        "Na"  =>  1,
    )

    fragment_basis_sets = Dict(
        ["O", "H"] => "aug-cc-pvtz",
        ["O", "H", "H", "H"] => "aug-cc-pvtz",
        ["Cl"] => "aug-cc-pvtz",
        ["Na"] => "aug-cc-pvtz"
    )
    _, cluster_labels, cluster_geoms = read_xyz(cluster_geom_file)
    
    rem_input_string_gas_phase = "\$rem
jobtype                 sp
method                  wB97M-V
unrestricted            1
basis                   mixed
xc_grid        2
scf_max_cycles          500
scf_convergence         6
thresh                  14
s2thresh 14
symmetry                0
sym_ignore              1
mem_total 256000
mem_static 16000
\$end"

    rem_input_string_with_env = "\$rem
jobtype                 sp
method                  wB97M-V
unrestricted            1
basis                   mixed
xc_grid        2
scf_max_cycles          500
scf_convergence         6
SCF_GUESS               read
thresh                  14
s2thresh 14
symmetry                0
sym_ignore              1
mem_total 256000
mem_static 16000
\$end"

    mkpath("qchem_input_files")
    @showprogress for i in eachindex(cluster_labels)
        cluster_charge = sum([atom_charges[label] for label in cluster_labels[i]])
        
        cluster = build_cluster(cluster_geoms[i], cluster_labels[i])
        
        # stores label, atom number, and basis set
        all_basis_sets = Tuple{String, Int, String}[]
        for i_frag in eachindex(cluster.indices)
            if haskey(fragment_basis_sets, cluster.labels[cluster.indices[i_frag]])
                for index in cluster.indices[i_frag]
                    push!(all_basis_sets, (cluster.labels[index], index, fragment_basis_sets[cluster.labels[cluster.indices[i_frag]]]))
                end
            else
                for index in cluster.indices[i_frag]
                    push!(all_basis_sets, (cluster.labels[index], index, "aug-cc-pvdz"))
                end
            end
        end

        basis_string = ""
        for i_atom in eachindex(all_basis_sets)
            basis_string = string(basis_string, all_basis_sets[i_atom][1], " ", all_basis_sets[i_atom][2], "\n", all_basis_sets[i_atom][3], "\n****\n")
        end

        geom_string = geometry_to_string(cluster_geoms[i], cluster_labels[i])
        # write anion file
        open(string("qchem_input_files/", infile_prefix, "_anion_sample_", i, ".in"), "w") do io
            write(io, "\$molecule\n")
            write(io, string(cluster_charge, " ", 1, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_gas_phase)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n@@@\n\n"))
            write(io, "\$molecule\n")
            write(io, string(cluster_charge, " ", 1, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_with_env)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n"))
            write(io, "\n\$external_charges\n")
            writedlm(io, readdlm(string("env_charges/charges_sample_", i, ".xyz")))
            write(io, "\$end\n\n")
        end

        # write radical file
        open(string("qchem_input_files/", infile_prefix, "_radical_sample_", i, ".in"), "w") do io
            write(io, "\$molecule\n")
            write(io, string(cluster_charge+1, " ", 2, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_gas_phase)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n@@@\n\n"))
            write(io, "\$molecule\n")
            write(io, string(cluster_charge+1, " ", 2, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_with_env)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n"))
            write(io, "\n\$external_charges\n")
            writedlm(io, readdlm(string("env_charges/charges_sample_", i, ".xyz")))
            write(io, "\$end\n\n")
        end
    end
end
