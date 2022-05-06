using Graphs
using Statistics
using StaticArrays
using ProgressBars

include("molecular_graph_utils.jl")

function count_rings(G::Graphs.SimpleGraph, ring_size::Int; return_paths::Bool=false, is_recursive_call::Bool=false, return_all_ring_counts_up_to_ring_size::Bool=false)
    visit_stack::Vector{Vector{Int}} = [[i] for i in vertices(G)]

    visited = Set{Int}()
    rings = Vector{StaticArrays.SVector{ring_size, Int}}()
    num_rings::Int = 0
    ring_key_set = Vector{Set{Int}}()
    smaller_rings = Vector{Set{Int}}()
    if !is_recursive_call
        for i in 3:(ring_size-1)
            num, small_rings_found = count_rings(G, i; return_paths=true, is_recursive_call=true)
            for j in 1:num
                push!(smaller_rings, Set(small_rings_found[j]))
            end
        end
    end

    while length(visit_stack) > 0
        path_to_expand::Vector{Int} = pop!(visit_stack)
        next_node::Int = path_to_expand[end]
        current_path_length::Int = length(path_to_expand)

        for neighbor in Graphs.neighbors(G, next_node)
            if current_path_length == ring_size
                if neighbor == path_to_expand[begin]
                    degrees::Vector{Int} = degree(induced_subgraph(G, path_to_expand)[1])
                    
                    # check if we've already found this ring
                    if mean(degrees) == 2 && !(Set(path_to_expand) in ring_key_set)
                        num_rings += 1
                        if return_paths
                            push!(rings, path_to_expand)
                        end
                        push!(ring_key_set, Set(path_to_expand))
                    end
                end
            elseif current_path_length < ring_size && !(neighbor in path_to_expand)
                new_path_to_expand = push!(copy(path_to_expand), neighbor)
                push!(visit_stack, new_path_to_expand)
            end
        end
    end
    
    # eliminate the graphs which are only one node different
    # than triplets of smaller rings sizes.
    # There is probably some more general rule here. Like if three rings share 
    # a single node then there is a ring around the perimeter which will be
    # erroneously counted.
    #
    # This is an abomination... But it works.
    if length(smaller_rings) >= 3
        for i in 1:length(smaller_rings)-2
            for j in (i+1):length(smaller_rings)-1
                for k in (j+1):length(smaller_rings)
                    if (length(smaller_rings[i]) < ring_size) && (length(smaller_rings[j]) < ring_size) && (length(smaller_rings[k]) < ring_size)
                        for i_ring in length(ring_key_set):-1:1
                            if length(setdiff(union(smaller_rings[i], smaller_rings[j], smaller_rings[k]), ring_key_set[i_ring])) == 1
                                num_rings -= 1
                                deleteat!(ring_key_set, i_ring)
                                if return_paths
                                    deleteat!(rings, i_ring)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    #if length(smaller_rings) >= 4
    #    for i in 1:length(smaller_rings)-3
    #        for j in (i+1):length(smaller_rings)-2
    #            for k in (j+1):length(smaller_rings)-1
    #                for l in (k+1):length(smaller_rings)
    #                    if (length(smaller_rings[i]) < ring_size) && (length(smaller_rings[j]) < ring_size) && (length(smaller_rings[k]) < ring_size) && (length(smaller_rings[k]) < ring_size)
    #                        for i_ring in length(ring_key_set):-1:1
    #                            if length(setdiff(union(smaller_rings[i], smaller_rings[j], smaller_rings[k], smaller_rings[l]), ring_key_set[i_ring])) == 1
    #                                num_rings -= 1
    #                                deleteat!(ring_key_set, i_ring)
    #                                deleteat!(rings, i_ring)
    #                            end
    #                        end
    #                    end
    #                end
    #            end
    #        end
    #    end
    #end

    if return_paths
        return num_rings, rings
    elseif return_all_ring_counts_up_to_ring_size
        ring_counts = zeros(Int, ring_size-2)
        for i in 1:length(smaller_rings)
            ring_counts[length(smaller_rings[i])-2] += 1
        end
        return ring_counts
    else
        return num_rings
    end
end

function count_rings(G::Graphs.SimpleGraph{Int}, ring_sizes::UnitRange{Int})
    counts::StaticArrays.MVector{length(ring_sizes), Int} = @MVector zeros(Int, length(ring_sizes))

    for (j, ring_size) in enumerate(ring_sizes)
        counts[j] = count_rings(G, ring_size)
    end
    return counts
end

function count_rings(graphs::AbstractVector{Graphs.SimpleGraph{Int}}, ring_sizes::UnitRange{Int})
    return pmap(x -> count_rings(x, ring_sizes), graphs)
end

function count_t1d_bonds(G::SimpleDiGraph{Int64})
    """
    Just counts the number of t1d bonds from a graph.
    """
    water_labels = label_water_type(G)

    num_t1d_pairs::Int = 0
    for i in vertices(G)
        if i in water_labels[:AAD]
            neighbor = outneighbors(G, i)[1]
            # if the neighbor is ADD, then we have a t1d pair
            if neighbor in water_labels[:ADD]
                num_t1d_pairs += 1
            end
        end
    end
    return num_t1d_pairs
end

function get_swb_labels(geom::Matrix{T}, labels::AbstractVector) where T <: Real
    """
    Forms digraph from geometry and then labels each ADD or AAD molecule as
    t1d, t1a, c2, c0, or c1a.
    """
    G = form_directed_molecular_graph(geom)
    # get the water labels and split everything into the monomers
    water_labels = label_water_type(G)
    monomers = get_array_of_waters(geom, labels)

    swb_labels = Dict{Symbol, Vector{Tuple{Int, Int}}}()

    for i in vertices(G)
        if i in water_labels[:AAD]
            neighbor = outneighbors(G, i)[1]
            # there is only one outneighbor. If it's AAD, then we have C2
            if neighbor in water_labels[:AAD]
                if :c2 in keys(swb_labels)
                    push!(swb_labels[:c2], (i, neighbor))
                else
                    swb_labels[:c2] = [(i, neighbor)]
                end
            # if the neighbor is ADD, then we have a t1d pair
            elseif neighbor in water_labels[:ADD]
                if :t1d in keys(swb_labels)
                    push!(swb_labels[:t1d], (i, neighbor))
                else
                    swb_labels[:t1d] = [(i, neighbor)]
                end
            # this doesn't fit in the taxonomy described by Kirov. So, we might re-classify this later
            else
                if :other in keys(swb_labels)
                    push!(swb_labels[:other], (i, neighbor))
                else
                    swb_labels[:other] = [(i, neighbor)]
                end
            end
        elseif i in water_labels[:ADD]
            # classify each dimer this double-dononor is a part of
            for neighbor in outneighbors(G, i)
                # this is a c0 dimer
                if neighbor in water_labels[:ADD]
                    if :c0 in keys(swb_labels)
                        push!(swb_labels[:c0], (i, neighbor))
                    else
                        swb_labels[:c0] = [(i, neighbor)]
                    end
                # this is either a t1a or c1a pair
                elseif neighbor in water_labels[:AAD]
                    donor_molecule = monomers[i]
                    acceptor_molecule = monomers[neighbor]
                    OHa_1 = acceptor_molecule[:,2] - acceptor_molecule[:,1]
                    OHa_2 = acceptor_molecule[:,3] - acceptor_molecule[:,1]
                    # the donor OH on the acceptor (which has a free OH) has a longer bond length
                    donor_vec_a = (length(OHa_1) > length(OHa_2)) ? OHa_2 : OHa_1

                    # now find which atom donates to the neighbor oxygen
                    HOd_1 = acceptor_molecule[:,1] - donor_molecule[:,2]
                    HOd_2 = acceptor_molecule[:,1] - donor_molecule[:,3]

                    non_donor_index_d = (length(HOd_1) < length(HOd_2)) ? 2 : 3

                    OH_non_donor = donor_molecule[:,non_donor_index_d] - donor_molecule[:,1]
                    OO_vec = acceptor_molecule[:,1] - donor_molecule[:,1]
                    # then these vectors point mostly in the same direction and are cis
                    dihedral = dihedral_angle(OH_non_donor, OO_vec, -donor_vec_a)
                    if dihedral <= 90.0 && dihedral >= -90.0
                        if :c1a in keys(swb_labels)
                            push!(swb_labels[:c1a], (i, neighbor))
                        else
                            swb_labels[:c1a] = [(i, neighbor)]
                        end
                    else
                        if :t1a in keys(swb_labels)
                            push!(swb_labels[:t1a], (i, neighbor))
                        else
                            swb_labels[:t1a] = [(i, neighbor)]
                        end
                    end
                # this also doesn't fit the Kirov taxonomy
                else
                    if :other in keys(swb_labels)
                        push!(swb_labels[:other], (i, neighbor))
                    else
                        swb_labels[:other] = [(i, neighbor)]
                    end
                end
            end
        else
            # this molecule isn't ADD or AAD so we just put it's dimers in other
            for neighbor in outneighbors(G, i)
                if :other in keys(swb_labels)
                    push!(swb_labels[:other], (i, neighbor))
                else
                    swb_labels[:other] = [(i, neighbor)]
                end
            end
        end
    end
    return swb_labels
end

function label_water_type(G::Graphs.SimpleDiGraph)
    """
    Takes a directed graph and determines the code for each node in the graph, e.g. AAD, ADD, etc.
    Stores the results in a dictionary indexed by the symbol :AAD, :ADD, etc.
    """
    labels = Dict{Symbol, Vector{Int}}()

    # Below is: (n_acceptor, n_donated) => label
    label_keys = Dict{Tuple{Int, Int}, Symbol}( 
                                                (0,0) => :None,
                                                (0,1) => :D, 
                                                (1,0) => :A,
                                                (0,2) => :DD,
                                                (2,0) => :AA,
                                                (1,1) => :AD,
                                                (2,1) => :AAD,
                                                (1,2) => :ADD,
                                                (2,2) => :AADD,
                                                (3,1) => :AAAD,
                                                (3,0) => :AAA,
                                                (3,2) => :AAADD,
                                                (4,1) => :AAAAD,
                                                (4,2) => :AAAADD)

    for i in vertices(G)
        n_accept::Int = length(inneighbors(G, i))
        n_donate::Int = length(outneighbors(G, i))
        label = label_keys[(n_accept, n_donate)]
        if label in keys(labels)
            push!(labels[label], i)
        else
            labels[label] = [i]
        end
    end
    return labels
end

function get_count_of_each_water_label(labels::Dict{Symbol, Vector{T}}) where T <: Real
    label_counts = Dict{Symbol, T}( :None => 0,
                                    :D => 0,
                                    :A => 0,
                                    :DD => 0,
                                    :AA => 0,
                                    :AD => 0,
                                    :AAD => 0,
                                    :ADD => 0,
                                    :AADD => 0,
                                    :AAAD => 0,
                                    :AAA => 0,
                                    :AAADD => 0,
                                    :AAAAD => 0,
                                    :AAAADD => 0)
    for key in keys(labels)
        label_counts[key] = length(labels[key])
    end
    return label_counts
end

function possible_water_labels()
    """
    This is just to get a consistent ordering of the labels for writing to a file.
    These labels are the h-bonding arrangement of a water molecule.
    """
    return Vector{Symbol}([:None, :D, :A, :DD, :AA, :AD, :AAD, :ADD, :AADD, :AAAD, :AAA, :AAADD, :AAAAD, :AAAADD])
end

function count_hydrogen_bonds(G::SimpleDiGraph{Int})
    return sum(outdegree.((G,), vertices(G)))
end

function count_free_OHs(G::SimpleDiGraph{Int})
    return 2 * length(vertices(G)) - count_hydrogen_bonds(G)
end

function OH_distances(geom::AbstractMatrix{T}) where T <: AbstractFloat
    OH_bond_lengths = []
    for i in 1:3:size(geom, 2)
        push!(OH_bond_lengths, norm(geom[:,i] - geom[:, i+1]))
        push!(OH_bond_lengths, norm(geom[:,i] - geom[:, i+2]))
    end
    return OH_bond_lengths
end
