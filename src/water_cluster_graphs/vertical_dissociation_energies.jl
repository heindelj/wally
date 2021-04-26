include("../molecule_tools/call_potential.jl")
include("molecular_graph_utils.jl")

function all_vertical_dissociations(potential::AbstractPotential, geom::Matrix{T}, labels::AbstractVector) where T <: Real
    """
    Presently, this assumes every molecule is a water (not necessarily sorted) and calculates
    the vertical dissociation energy of each molecule (cluster energy minus energy without one monomer.)
    """
    monomers = get_array_of_waters(geom, labels)

    total_energy::T = get_energy(potential, hcat(monomers...))
    vertical_dissociation_energies::Vector{T} = zeros(T, length(monomers))

    for i in 1:length(monomers)
        if i == 1
            vertical_dissociation_energies[i] = total_energy - get_energy(potential, hcat(monomers[2:end]...))
        elseif i == length(monomers)
            vertical_dissociation_energies[i] = total_energy - get_energy(potential, hcat(monomers[1:end-1]...))
        else
            vertical_dissociation_energies[i] = total_energy - get_energy(potential, hcat(monomers[1:(i-1)]..., monomers[(i+1):end]...))
        end
    end
    return vertical_dissociation_energies
end

function get_swb_labels(geom::Matrix{T}, labels::AbstractVector)
    """
    Forms digraph from geometry and then labels each ADD or AAD molecule as
    t1d, t1a, c2, c0, or c1a.
    """
    G = form_directed_molecular_graph(geom)
    # get the water labels and split everything into the monomers
    water_labels = label_water_type(G)
    monomers = get_array_of_waters(geom, labels)

    for i in vertices(G)
        # just write the code Joe
    end
end