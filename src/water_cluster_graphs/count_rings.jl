using LightGraphs
using Statistics
using StaticArrays
using ProgressBars

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
    counts::Vector{StaticArrays.MVector{length(ring_sizes), Int}} = [@MVector zeros(Int, length(ring_sizes)) for i in 1:length(graphs)]

    Threads.@threads for i in ProgressBar(1:length(graphs))
        counts[i] = count_rings(graphs[i], ring_sizes)
    end
    return counts
end
