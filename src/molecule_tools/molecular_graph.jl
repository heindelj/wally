using Graphs
include("molecular_cluster.jl")
include("water_tools.jl")

struct MolecularGraph
    G::SimpleGraph
    coords::Matrix{Float64}
    labels::Vector{String}
end

"""
Constructs a molecular graph by calling the is_bonded function which
determines connectivity based on covalent radii.
"""
function build_molecular_graph(coords::Matrix{Float64}, labels::Vector{String})
    adj_matrix = zeros(length(labels), length(labels))
    for i in 1:(length(labels)-1)
        for j in (i+1):length(labels)
            if is_bonded(norm(coords[:, i] - coords[:, j]), labels[i], labels[j])
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
            end
        end
    end
    G = SimpleGraph(adj_matrix)
    return MolecularGraph(G, coords, labels)
end

"""
Constructs a molecular graph treating this as an ion-water cluster. So, we will sort
the cluster and then group everything into fragments of water, OH-, H3O+, and
monoatomic ions. We will then use various h-bond definitions to construct the
connectivity of the graph.
"""
function build_noncovalent_molecular_graph(coords::Matrix{Float64}, labels::Vector{String})
    sorted_labels, sorted_coords = sort_water_cluster(coords, labels)
    
    

    #adj_matrix = zeros(length(labels), length(labels))
    #for i in 1:(length(labels)-1)
    #    for j in (i+1):length(labels)
    #        if is_bonded(norm(coords[:, i] - coords[:, j]), labels[i], labels[j])
    #            adj_matrix[i, j] = 1
    #            adj_matrix[j, i] = 1
    #        end
    #    end
    #end
    G = SimpleGraph(adj_matrix)
    return MolecularGraph(G, coords, labels)
end
