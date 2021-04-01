include("../../molecule_tools/water_tools.jl")
include("../../molecule_tools/read_xyz.jl")
using LightGraphs
using ProgressBars

function adjacency_matrix(water_cluster_geom::Array{T, 2}) where T <: AbstractFloat
    """
    Computes the adjacency matrix of a water cluster geometry based on the r_psi_hbond criterion.
    Expects the molecules in OHH order.
    """
    adj_matrix = zeros(Int, size(water_cluster_geom, 2) ÷ 3, size(water_cluster_geom, 2) ÷ 3)
    hbonds = r_psi_hydrogen_bonds(water_cluster_geom)
    for (h_donor_idx, o_acceptor_idx) in hbonds
        adj_matrix[(h_donor_idx - 1) ÷ 3 + 1, o_acceptor_idx ÷ 3 + 1] = 1
    end
    adj_matrix += adj_matrix'
    return adj_matrix
end

function adjacency_matrix(water_cluster_geoms::Array{Array{T, 2}, 1}) where T <: AbstractFloat
    """
    Computes the adjacency matrices of many water cluster geometries based on the r_psi_hbond criterion.
    Expects the molecules in OHH order.
    """
    adj_matrices = []
    for geom in water_cluster_geoms
        push!(adj_matrices, adjacency_matrix(geom))
    end
    return adj_matrices
end

function form_molecular_graph(water_cluster_geom::Array{T, 2}) where T <: AbstractFloat
    """
    Makes LightGraph out of a molecular structure by computing the structures adjacency matrix.
    """
    return LightGraphs.SimpleGraph(adjacency_matrix(water_cluster_geom))
end

function form_molecular_graph(water_cluster_geoms::Array{Array{T, 2}, 1}) where T <: AbstractFloat
    """
    Makes LightGraph out of a molecular structure by computing the structures adjacency matrix.
    """
    return LightGraphs.SimpleGraph.(adjacency_matrix(water_cluster_geoms))
end

function split_clusters_into_families(water_cluster_geoms::Array{Array{T, 2}, 1}) where T <: AbstractFloat
    """
    Takes a set of water cluster geometries and splits them into families based on their
    connectivity. That is, we compute all of the adjacency matrices and then form a graph
    and do an isomorphism check with known families. If there's no match this cluster 
    begins a new family. Only the indices are stored to avoid copying the geometries.
    """
    indices_into_families::Array{Array{Int}} = [[1]] # there will always be one family
    println("Forming molecular graphs...")
    graphs = form_molecular_graph(water_cluster_geoms)
    println("Breaking graphs into families based on isomorphism...")
    for i in ProgressBar(2:length(graphs))
        has_matched::Bool = false
        for families in indices_into_families
            if !has_matched
                if LightGraphs.Experimental.has_isomorph(graphs[i], graphs[families[begin]])
                    push!(families, i)
                    has_matched = true
                    break
                end
            end
        end
        # if you haven't matched, then this is a new family
        if !has_matched
            push!(indices_into_families, [i])
        end
    end
    return indices_into_families
end

function write_cluster_families(header::Array{String,1}, labels::Array{Array{String,1},1}, geoms::Array{Array{T, 2}, 1}, indices_into_families::Array{Array{Int}}; output_directory::AbstractString="") where T <: AbstractFloat
    """
    Writes out all of the structures for each family passed in, each family
    in its own file. I name the file based on number of waters (more generic)
    and the lowest energy in the family.
    """
    for (i_family, family_indices) in enumerate(indices_into_families)
        family_headers = view(header, family_indices)

        # find the lowest energy in this family for output purposes
        N_atoms = 0
        lowest_energy = maxintfloat(Float64)
        for family_header in family_headers
            tokens = split(family_header)
            N = parse(Int, tokens[begin])
            energy = parse(Float64, tokens[end])
            N_atoms = (N > N_atoms) ? N : N_atoms 
            lowest_energy = (lowest_energy < energy) ? lowest_energy : energy
        end

        ofile = string("W", N_atoms ÷ 3, "_", lowest_energy, "_kcal_family_", i_family, ".xyz")

        family_labels = view(labels, family_indices)
        family_geoms = view(geoms, family_indices)
        write_xyz(ofile, family_headers, family_labels, family_geoms, directory=output_directory)
    end
end