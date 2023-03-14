using Rotations, Sobol, Distributions
include("molecular_axes.jl")
include("vdw_radii.jl")

"""
Takes two fragments and a direction vector (we normalize it to be sure),
and find the shortest van der waal's contact between the fragments along
this direction vector. This is useful when trying to choose an appropriate
distance to place the fragments from one another when they have random orientations.
"""
function find_shortest_vdw_contact_distance(
    fragment_1::AbstractMatrix{Float64}, labels_1::Vector{String},
    fragment_2::AbstractMatrix{Float64}, labels_2::Vector{String},
    direction::Vector{Float64}, dR_min::Float64=-0.75, dR_max::Float64=0.6
)
    direction = normalize(direction)
    atom_pairs = Tuple{String,String}[]
    for i in eachindex(labels_1)
        for j in eachindex(labels_2)
            push!(atom_pairs, (labels_1[i], labels_2[j]))
        end
    end

    # notice we will include both orderings of the tuple
    # this is fine and makes life easier
    unique_atom_pairs = Set(atom_pairs)
    vdw_contact_distances = Dict{Tuple{String,String},Float64}()
    for pair in unique_atom_pairs
        vdw_contact_distances[pair] = (vdw_radius(pair[1]) + vdw_radius(pair[2]))
        vdw_contact_distances[(pair[2], pair[1])] = (vdw_radius(pair[1]) + vdw_radius(pair[2]))
    end


    frag_1 = copy(fragment_1)
    frag_2 = copy(fragment_2)
    rand_distance = rand(Uniform(dR_min, dR_max), 1)[1]

    # start by identifying the pair which will make contact first
    R0 = 100.0
    R0_last = 99.0
    for i in eachindex(eachcol(fragment_2))
        @views frag_2[:, i] = fragment_2[:, i] + R0 * direction
    end
    shortest_distance = 10000000000.0
    closest_pair = (0, 0)
    for i in eachindex(eachcol(frag_1))
        for j in eachindex(eachcol(frag_2))
            distance = norm(fragment_1[:, i] - frag_2[:, j]) - vdw_contact_distances[(labels_1[i], labels_2[j])] + rand_distance
            if distance < shortest_distance
                shortest_distance = distance
                closest_pair = (i, j)
            end
        end
    end

    # start with a large separation and walk in by half the shortest
    # distance until the atoms make contact.
    max_iter = 1000
    iter = 1
    while iter < max_iter
        for i in eachindex(eachcol(fragment_2))
            @views frag_2[:, i] = fragment_2[:, i] + R0 * direction
        end
        if (norm(fragment_1[:, closest_pair[1]] - (fragment_2[:, closest_pair[2]] + R0 * direction)) - vdw_contact_distances[(labels_1[closest_pair[1]], labels_2[closest_pair[2]])] + rand_distance) < 1e-3
            short_deviation = abs(norm(fragment_1[:, closest_pair[1]] - (fragment_2[:, closest_pair[2]] + R0 * direction)) - vdw_contact_distances[(labels_1[closest_pair[1]], labels_2[closest_pair[2]])] + rand_distance)
            long_deviation = abs(norm(fragment_1[:, closest_pair[1]] - (fragment_2[:, closest_pair[2]] + R0_last * direction)) - vdw_contact_distances[(labels_1[closest_pair[1]], labels_2[closest_pair[2]])] + rand_distance)
            w_short = short_deviation / (long_deviation + short_deviation)
            w_long = long_deviation / (long_deviation + short_deviation)
            R0 = w_short * R0 + w_long * R0_last
            return R0, vdw_contact_distances[(labels_1[closest_pair[1]], labels_2[closest_pair[2]])] + rand_distance
        end
        R0_last = R0
        R0 -= 0.5 * (norm(fragment_1[:, closest_pair[1]] - (fragment_2[:, closest_pair[2]] + R0 * direction)) - vdw_contact_distances[(labels_1[closest_pair[1]], labels_2[closest_pair[2]])] + rand_distance)
        iter += 1
    end
    return R0, vdw_contact_distances[(labels_1[closest_pair[1]], labels_2[closest_pair[2]])]
end

function get_random_dimer_geometries_along_direction(
    fragment_1::AbstractMatrix{Float64}, labels_1::Vector{String},
    fragment_2::AbstractMatrix{Float64}, labels_2::Vector{String},
    direction::Vector{Float64},
    num_geoms::Int = 5, dR_min::Float64 = -0.75, dR_max::Float64 = 0.6
)

    sampled_geoms = [zeros(3, size(fragment_1, 2) + size(fragment_2, 2)) for _ in 1:num_geoms]
    sampled_labels = [vcat(labels_1, labels_2) for _ in 1:num_geoms]
    for i in 1:num_geoms
        t, _ = find_shortest_vdw_contact_distance(fragment_1, labels_1, fragment_2, labels_2, direction, dR_min, dR_max)
        frag_2 = copy(fragment_2)
        for j in eachindex(eachcol(frag_2))
            @views frag_2[:, j] = fragment_2[:, j] + t * direction
        end

        sampled_geoms[i] = hcat(fragment_1, frag_2)
    end

    return sampled_labels, sampled_geoms
end

"""
Samples random orientations of two fragments based on a Sobol
sequence and then gets some number of geometries randomly
distributed within dR_min and dR_max around the vdw contact
distance of the two molecules.
"""
function sample_psuedorandom_dimers(
    fragment_1::AbstractMatrix{Float64}, labels_1::Vector{String},
    fragment_2::AbstractMatrix{Float64}, labels_2::Vector{String},
    num_geoms_total::Int=4000, num_geoms_per_direction::Int=5,
    dR_min::Float64 = -0.75, dR_max::Float64 = 0.6; num_to_skip::Int=0
)
    # ensure fragment centers of mass are at the origin
    frag_1 = copy(fragment_1)
    frag_2 = copy(fragment_2)
    com_1 = center_of_mass(frag_1, labels_1)
    for i in eachindex(eachcol(frag_1))
        @views frag_1[:, i] -= com_1
    end
    com_2 = center_of_mass(frag_2, labels_2)
    for i in eachindex(eachcol(frag_2))
        @views frag_2[:, i] -= com_2
    end

    # get pseudorandom direction vectors from Sobol sequence
    seq = skip(SobolSeq([-1, -1, -1], [1, 1, 1]), num_to_skip, exact=true)
    all_labels = Vector{String}[]
    all_geoms  = Matrix{Float64}[]
    for i in 1:(num_geoms_total ÷ num_geoms_per_direction)
        direction = next!(seq)
        if norm(direction) == 0.0
            direction = next!(seq)
        end
        normalize!(direction)
        R = rand(RotMatrix{3}) # get random orientation
        sampled_labels, sampled_geoms = get_random_dimer_geometries_along_direction(
            frag_1, labels_1, R * frag_2, labels_2, direction,
            num_geoms_per_direction, dR_min, dR_max
        )
        append!(all_labels, sampled_labels)
        append!(all_geoms, sampled_geoms)
    end
    return all_labels, all_geoms
end

"""
Same as above except a collection is passed in for each fragment
and we randomly sample one of these each time.
"""
function sample_psuedorandom_dimers(
    fragment_1::AbstractVector{Matrix{Float64}}, labels_1::Vector{String},
    fragment_2::AbstractVector{Matrix{Float64}}, labels_2::Vector{String},
    num_geoms_total::Int=4000, num_geoms_per_direction::Int=5,
    dR_min::Float64=-0.75, dR_max::Float64=0.6; num_to_skip::Int=0
)
    all_labels = Vector{String}[]
    all_geoms = Matrix{Float64}[]
    for i in 1:(num_geoms_total ÷ num_geoms_per_direction)
        frag_1 = fragment_1[rand(1:length(fragment_1))]
        frag_2 = fragment_2[rand(1:length(fragment_2))]
        sampled_labels, sampled_geoms = sample_psuedorandom_dimers(
            frag_1, labels_1, frag_2, labels_2,
            num_geoms_per_direction, num_geoms_per_direction,
            dR_min, dR_max, num_to_skip=(num_to_skip + i)
        )
        append!(all_labels, sampled_labels)
        append!(all_geoms, sampled_geoms)
    end
    return all_labels, all_geoms
end