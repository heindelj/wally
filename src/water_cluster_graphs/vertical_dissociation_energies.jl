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

function vertical_dissociations(potential::AbstractPotential, geom::Matrix{T}, labels::AbstractVector, excluded_indices::Union{Nothing, Vector{Tuple{Int, Int}}}=nothing) where T <: Real
    """
    Presently, this assumes every molecule is a water (not necessarily sorted) and calculates
    the vertical dissociation energy of each molecule (cluster energy minus energy without one monomer.)
    If excluded_indices is provided, then vertical dissociation excluding each of these indices will be calculated.
    """
    monomers = get_array_of_waters(geom, labels)

    total_energy::T = get_energy(potential, hcat(monomers...))
    vertical_dissociation_energies::Vector{T} = zeros(T, 0)

    if excluded_indices === nothing
        vertical_dissociation_energies = zeros(T, length(monomers))
    else
        vertical_dissociation_energies = zeros(T, length(excluded_indices))
    end

    if excluded_indices === nothing
        for i in 1:length(monomers)
            if i == 1
                vertical_dissociation_energies[i] = total_energy - get_energy(potential, hcat(monomers[2:end]...))
            elseif i == length(monomers)
                vertical_dissociation_energies[i] = total_energy - get_energy(potential, hcat(monomers[1:end-1]...))
            else
                vertical_dissociation_energies[i] = total_energy - get_energy(potential, hcat(monomers[1:(i-1)]..., monomers[(i+1):end]...))
            end
        end
    else
        for (i, idx_tuple) in enumerate(excluded_indices)
            indices = [1:length(monomers)...]
            vertical_dissociation_energies[i] = total_energy - get_energy(potential, hcat(monomers[deleteat!(indices, sort!([idx_tuple...]))]...))
        end
    end
    return vertical_dissociation_energies
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
