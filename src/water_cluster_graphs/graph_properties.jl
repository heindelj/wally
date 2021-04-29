using LightGraphs
using Statistics
using StaticArrays
using ProgressBars

include("molecular_graph_utils.jl")

function count_rings(G::LightGraphs.SimpleGraph, ring_size::Int; return_paths::Bool=false)
    visit_stack::Vector{Vector{Int}} = [[i] for i in vertices(G)]

    visited = Set{Int}()
    rings = Vector{StaticArrays.SVector{ring_size, Int}}()
    num_rings::Int = 0
    ring_key_set = Vector{Set{Int}}()

    while length(visit_stack) > 0
        path_to_expand::Vector{Int} = pop!(visit_stack)
        next_node::Int = path_to_expand[end]
        current_path_length::Int = length(path_to_expand)

        for neighbor in LightGraphs.neighbors(G, next_node)
            if current_path_length == ring_size
                if neighbor == path_to_expand[begin]
                    G_sub = induced_subgraph(G, path_to_expand)
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
    if return_paths
        return num_rings, rings
    end
    return num_rings
end

function count_rings(G::LightGraphs.SimpleGraph{Int}, ring_sizes::UnitRange{Int})
    counts::StaticArrays.MVector{length(ring_sizes), Int} = @MVector zeros(Int, length(ring_sizes))

    for (j, ring_size) in enumerate(ring_sizes)
        counts[j] = count_rings(G, ring_size)
    end
    return counts
end

function count_rings(graphs::AbstractVector{LightGraphs.SimpleGraph{Int}}, ring_sizes::UnitRange{Int})
    return pmap(x -> count_rings(x, ring_sizes), graphs)
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

function count_hydrogen_bonds(G::SimpleDiGraph{Int})
    return sum(outdegree.((G,), vertices(G)))
end

function count_free_OHs(G::SimpleDiGraph{Int})
    return 2 * length(vertices(G)) - count_hydrogen_bonds(G)
end