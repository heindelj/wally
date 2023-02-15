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
    centers::Vector{SVector{3, Float64}} # location of each molecule
    indices::Vector{Vector{Int}} # maps from molecule indices to full system indices
    geom::Matrix{Float64} # full system coordinates
    labels::Vector{String} # full system labels
end

function is_bonded(distance::Float64, label1::String, label2::String)
    return distance < (covalent_radius(label1) + covalent_radius(label2) + 0.4)
end

function find_bonded_atoms!(
        neighboring_indices::Vector{Vector{Int}},
        neighbor_dists::Vector{Vector{Float64}},
        molecular_indices::Vector{Int},
        starting_index::Int,
        labels::Vector{String},
        max_neighbors::Int=8)
    """
    neighboring_indices and neighbor_dists are vectors of indices
    and distances to that index for all atoms. The indices are sorted
    against the distance but we don't utilize that fact.

    We use the above arrays to find all bonded neighbors of a particular
    atom and then recursively follow the neighbors until the only neighbor
    of each atom is itself or an atom contained in the list already.
    """
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

function build_cluster(geom::AbstractMatrix{Float64}, labels::AbstractVector{String}, max_number_of_atoms::Int=8)
    """
    Default method for building a molecular cluster. Uses covalent radius definition of bonding and centroid as position of molecular subunits.
    """
    nl = KDTree(geom)
    indices, dists = knn(nl, geom, length(labels) > max_number_of_atoms+1 ? max_number_of_atoms+1 : length(labels), true)

    cluster_indices = Vector{Vector{Int}}()
    for i in 1:length(labels)
        if length(cluster_indices) == 0
            molecular_indices = Int[]
            find_bonded_atoms!(indices, dists, molecular_indices, i, labels)
            push!(cluster_indices, molecular_indices)
        elseif sum(i .∈ cluster_indices) == 0
            molecular_indices = Int[]
            find_bonded_atoms!(indices, dists, molecular_indices, i, labels)
            push!(cluster_indices, molecular_indices)
        end
    end
    cluster_centers = [SVector{3, Float64}(centroid(geom[:,index_set])) for index_set in cluster_indices]
    return Cluster(cluster_centers, cluster_indices, geom, labels)
end

function find_n_nearest_neighbors(cluster::Cluster, center_indices::Vector{Int}, n::Int, sortres::Bool=true)
    """
    Finds the n nearest neighbors for a collection of indices.
    The center_indices are the indices for the cluster, not the geometry
    from which the cluster is derived. So, the easiest way to get these
    indices is by looking up based on the molecular formula.
    """
    nl = KDTree(cluster.centers)
    neighbor_indices, _ = knn(nl, cluster.centers[center_indices], n+1, sortres) # the self of something is counted as a neighbor so add one to number requested
    labels_out = Vector{Vector{String}}()
    geoms_out = Vector{Matrix{Float64}}()
    for i in 1:length(neighbor_indices)
        total_indices = reduce(vcat, cluster.indices[neighbor_indices[i]])
        push!(labels_out, cluster.labels[total_indices])
        push!(geoms_out, cluster.geom[:, total_indices])
    end
    return labels_out, geoms_out
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
    geoms_out  = [Matrix{Float64}[] for _ in 1:Threads.nthreads()]
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