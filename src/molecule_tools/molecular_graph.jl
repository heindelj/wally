using Graphs
include("molecular_cluster.jl")

struct MolecularGraph
    G::SimpleGraph
    coords::Matrix{Float64}
    labels::Vector{String}
end

function build_molecular_graph(coords::Matrix{Float64}, labels::Vector{String})
    """
    Constructs a molecular graph by calling the is_bonded function which
    determines connectivity based on covalent radii. Other options are
    possible such as hydrogen-bonding, but those are not supported yet.
    """
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
