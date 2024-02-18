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
    
    cluster = build_cluster(sorted_coords, sorted_labels)

    # Currently we don't have a generic definiton of an ion-water
    # hydrogen bond. In lieu of this, we just look at the water
    # h-bond network and return that as the graph. This will work
    # fine when there is only a single ion, but in the future
    # this should be made more robust with a proper definition
    # of ion-water hydrogen bonds.
    # Idea: Use the parallel polarizability, just like Martin did
    # to come up with a generic definition of hydrogen bonds.
    # Probably there is some similar maximum in the polarizability
    # as the ion-water hydrogen bond breaks.

    water_labels = String[]
    water_indices = Int[]
    for i in eachindex(cluster.indices)
        if length(cluster.indices[i]) == 3
            append!(water_labels, cluster.labels[cluster.indices[i]])
            append!(water_indices, cluster.indices[i])
        end
    end

    water_coords = sorted_coords[:, water_indices]

    hbonds = r_psi_hydrogen_bonds(water_coords)

    adj_matrix = zeros(Int, length(water_labels) ÷ 3, length(water_labels) ÷ 3)
    for key in keys(hbonds)
        donor_key = ((key-1) ÷ 3) + 1
        acceptor_value = ((hbonds[key]-1) ÷ 3) + 1
        adj_matrix[donor_key, acceptor_value] = 1
        adj_matrix[acceptor_value, donor_key] = 1
    end
    G = SimpleGraph(adj_matrix)
    return MolecularGraph(G, water_coords, water_labels)
end
