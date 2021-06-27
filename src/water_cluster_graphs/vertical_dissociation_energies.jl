include("../molecule_tools/call_potential.jl")
include("molecular_graph_utils.jl")

function get_dimer_energies(potential::AbstractPotential, geom::Matrix{T}, labels::AbstractVector, dimer_indices::Vector{Tuple{Int, Int}}) where T <: Real
    """
    Get the energy of each watwer dimer specified by dimer_indices.
    """
    monomers = get_array_of_waters(geom, labels)

    dimer_energies::Vector{T} = zeros(T, length(dimer_indices))
    for i in 1:length(dimer_indices)
        dimer_energies[i] = get_energy(potential, hcat(monomers[[dimer_indices[i]...]]...))
    end
    return dimer_energies
end

function vertical_dissociations(potential::AbstractPotential, geom::Matrix{T}, labels::AbstractVector, excluded_indices::Vector{Tuple{Int, Int}}) where T <: Real
    """
    Presently, this assumes every molecule is a water (not necessarily sorted) and calculates
    the vertical dissociation energy of each molecule (cluster energy minus energy without one monomer.)
    If excluded_indices is provided, then vertical dissociation excluding each of these indices will be calculated.
    """
    monomers = get_array_of_waters(geom, labels)

    total_energy::T = get_energy(potential, hcat(monomers...))
    vertical_dissociation_energies::Vector{T} = zeros(T, 0)

    vertical_dissociation_energies = zeros(T, length(excluded_indices))

    for (i, idx_tuple) in enumerate(excluded_indices)
        indices = [1:length(monomers)...]
        vertical_dissociation_energies[i] = total_energy - get_energy(potential, hcat(monomers[deleteat!(indices, sort!([idx_tuple...]))]...)) - get_energy(potential, hcat(monomers[[idx_tuple...]]...))
    end
    return vertical_dissociation_energies
end

