using StaticArrays, NearestNeighbors, ProgressBars
using StatsBase: countmap
include("covalent_radii.jl")
include("molecular_axes.jl")
include("read_xyz.jl")

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
function find_n_nearest_neighbors(cluster::Cluster, center_indices::Vector{Int}, n::Int, sortres::Bool=true)
    nl = KDTree(cluster.centers)
    neighbor_indices, _ = knn(nl, cluster.centers[center_indices], n + 1, sortres) # the self of something is counted as a neighbor so add one to number requested
    labels_out = Vector{Vector{String}}()
    geoms_out = Vector{Matrix{Float64}}()
    for i in 1:length(neighbor_indices)
        total_indices = reduce(vcat, cluster.indices[neighbor_indices[i]])
        push!(labels_out, cluster.labels[total_indices])
        push!(geoms_out, cluster.geom[:, total_indices])
    end
    return labels_out, geoms_out
end

"""
Finds the n nearest neighbors for an index.
The center_index is the index for the cluster, not the geometry
from which the cluster is derived. So, the easiest way to get the
index is by looking up based on the molecular formula.
Range is a distance in the same units as that of the cluster.
Excluded indices will the centers in the provided array.
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

function find_n_nearest_neighbors(cluster::Cluster, chemical_formula::Vector{String}, n::Int, sortres::Bool=true)
    molecule_indices = molecules_by_formula(cluster, chemical_formula)
    return find_n_nearest_neighbors(cluster, molecule_indices, n, sortres)
end

function write_n_nearest_neighbors(geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, chemical_formula::Vector{String}, n::Int, file_name::String="subclusters.xyz")
    labels_out = [Vector{String}[] for _ in 1:Threads.nthreads()]
    geoms_out = [Matrix{Float64}[] for _ in 1:Threads.nthreads()]
    Threads.@threads for i in ProgressBar(1:length(geoms))
        id = Threads.threadid()
        cluster = build_cluster(geoms[i], labels[i])
        labels_frame, geoms_frame = find_n_nearest_neighbors(cluster, chemical_formula, n)
        append!(labels_out[id], labels_frame)
        append!(geoms_out[id], geoms_frame)

        if (i % 200) == 0
            final_labels_out = Vector{String}[]
            final_geoms_out = Matrix{Float64}[]
            append!.((final_labels_out,), labels_out)
            append!.((final_geoms_out,), geoms_out)
            write_xyz(file_name, [string(length(final_labels_out[j]), "\n") for j in 1:length(final_labels_out)], final_labels_out, final_geoms_out)
        end
    end
    final_labels_out = Vector{String}[]
    final_geoms_out = Matrix{Float64}[]
    append!.((final_labels_out,), labels_out)
    append!.((final_geoms_out,), geoms_out)
    write_xyz(file_name, [string(length(final_labels_out[j]), "\n") for j in 1:length(final_labels_out)], final_labels_out, final_geoms_out)
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
in the cluster. The sample will be expanded
"""
function sample_random_clusters_within_range(
    geoms::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    chemical_formula::Vector{String},
    radius::Float64,
    expand_sample_if_fragment_found::Vector{Vector{String}},
    expansion_radius::Float64,
    skip_first_n_frames::Int
)
    cluster = build_cluster(geoms[1], labels[1])
    center_indices = molecules_by_formula(cluster, chemical_formula)
    neighbor_indices = find_neighbors_within_range(cluster, center_indices[1], radius, false)
    fragmented_labels = [cluster.labels[cluster.indices[i]] for i in neighbor_indices]
    fragmented_geom = [cluster.geom[:, cluster.indices[i]] for i in neighbor_indices]
    
    # TODO: add in the random sampling of the frames that respects
    # how many frames we want to skip!

    already_expanded = Int[] # stores expansion indices we already dealt with
    @label recursively_expand
    had_to_expand = false
    for i in eachindex(expand_sample_if_fragment_found)
        label_index = findall(x->x==expand_sample_if_fragment_found[i], fragmented_labels)[1]
        expansion_center_index = neighbor_indices[label_index]
        if label_index !== nothing && expansion_center_index ∉ already_expanded
            had_to_expand = true
            push!(already_expanded, expansion_center_index)
            expanded_indices = find_neighbors_within_range(cluster, expansion_center_index, expansion_radius)
            expanded_indices = setdiff(expanded_indices, neighbor_indices)

            expanded_labels   = [cluster.labels[cluster.indices[i]] for i in setdiff(expanded_indices, neighbor_indices)]
            expanded_geom     = [cluster.geom[:, cluster.indices[i]] for i in setdiff(expanded_indices, neighbor_indices)]
            fragmented_labels = [fragmented_labels..., expanded_labels...]
            fragmented_geom   = [fragmented_geom..., expanded_geom...]
        end
        if had_to_expand
            @goto recursively_expand
        end
    end

    # Get the other labels as well for the environment
    return reduce(vcat, fragmented_labels), reduce(hcat, fragmented_geom)
end