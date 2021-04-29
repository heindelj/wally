include("../molecule_tools/call_potential.jl")
include("molecular_graph_utils.jl")

function dihedral_angle(v1::AbstractVector, v2::AbstractVector, v3::AbstractVector)
    """
    Computes the dihedral angle from three vectors in degrees.
    See this SE question for diagram and math: https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    """
    n1 = cross(v1 / norm(v1), v2 / norm(v2))
    n1 /= norm(n1)
    n2 = cross(v2 / norm(v2), v3 / norm(v3))
    n2 /= norm(n2)
    m1 = cross(n1, v2 / norm(v2))

    x = n1 ⋅ n2
    y = m1 ⋅ n2

    return atand(y, x)
end

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

