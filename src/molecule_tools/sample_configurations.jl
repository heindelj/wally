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
        radius_1 = ionic_radius(pair[1])
        radius_2 = ionic_radius(pair[2])
        if radius_1 == 0.0
            radius_1 = vdw_radius(pair[1])
        end
        if radius_2 == 0.0
            radius_2 = vdw_radius(pair[2])
        end

        vdw_contact_distances[pair] = radius_1 + radius_2
        vdw_contact_distances[(pair[2], pair[1])] = radius_1 + radius_2
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
Finds a t value to move along a vector such that the closest contact is between
R_inner and R_outer. I just do it by binary search.
"""
function find_t_between_radii(
    fragment_1::AbstractMatrix{Float64}, fragment_2::AbstractMatrix{Float64},
    direction::Vector{Float64}, R_inner::Float64, R_outer::Float64, max_iter=1000
)
    t_last = 15.0
    t_current = 10.0
    step_size = 0.5
    atom_too_close = false
    atom_at_right_distance = false

    frag_2 = copy(fragment_2)
    for _ in 1:max_iter
        # update positions of all atoms
        for i in eachindex(eachcol(fragment_2))
            @views frag_2[:, i] = fragment_2[:, i] + t_current * direction
        end

        # check for shortest inter-atomic distance
        for i in 1:size(fragment_1, 2)
            for j in 1:size(frag_2, 2)
                r_ij = norm(fragment_1[:,i] - frag_2[:,j])
                if r_ij < R_inner
                    atom_too_close = true
                    break
                elseif r_ij > R_inner && r_ij < R_outer
                    atom_at_right_distance = true
                end
            end
            if atom_too_close
                break
            end
        end

        ### propose new t based on outcome of distance checks ###
        # If there is an atom too close, make t larger by choosing midpoint
        # current and previous t. Don't update previous t.
        if atom_too_close
            atom_too_close = false
            t_current += 0.5 * (t_last - t_current)
            continue
        end

        # Now check if there was an atom in the right distance.
        # If so, then all atoms are at the right distance or farther.
        # Then we're done.
        if atom_at_right_distance
            return t_current
        end

        # If all atoms are still too far away, then t_current is safely beyond
        # contact and we shrink t_current by step_size.
        t_last = copy(t_current)
        t_current -= step_size
        atom_too_close = false
        atom_at_right_distance = false
    end
    @warn "Failed to get atoms within two radii. Just returning what we have."
    return t_current
end

function get_random_dimer_geometries_along_direction_between_spheres(
    fragment_1::AbstractMatrix{Float64}, labels_1::Vector{String},
    fragment_2::AbstractMatrix{Float64}, labels_2::Vector{String},
    direction::Vector{Float64}, R_inner::Float64, R_outer::Float64,
    num_geoms::Int = 5
)
    @assert R_inner < R_outer "Inner radius is larger than outer radius. What are you asking for?"

    sampled_geoms = [zeros(3, size(fragment_1, 2) + size(fragment_2, 2)) for _ in 1:num_geoms]
    sampled_labels = [vcat(labels_1, labels_2) for _ in 1:num_geoms]
    # NOTE: We randomly rotate both fragments in case one of them is an atom.
    # Just to make sure we get the random orientation.
    for i in 1:num_geoms
        R = rand(RotMatrix{3}) # get random orientation
        frag_2 = R * copy(fragment_2)
        t = find_t_between_radii(R * fragment_1, frag_2, direction, R_inner, R_outer)
        for j in eachindex(eachcol(frag_2))
            @views frag_2[:, j] = R * fragment_2[:, j] + t * direction
        end

        sampled_geoms[i] = hcat(R * fragment_1, frag_2)
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
        direction = Sobol.next!(seq)
        if norm(direction) == 0.0
            direction = Sobol.next!(seq)
        end
        normalize!(direction)
        sampled_labels, sampled_geoms = get_random_dimer_geometries_along_direction(
            frag_1, labels_1, frag_2, labels_2, direction,
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

function sample_psuedorandom_dimers_in_spheres(
    fragment_1::AbstractMatrix{Float64}, labels_1::Vector{String},
    fragment_2::AbstractMatrix{Float64}, labels_2::Vector{String},
    R_min::Float64, R_max::Float64,
    num_geoms_total::Int=4000, num_geoms_per_direction::Int=5; num_to_skip::Int=0
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

    radius_step_size = (R_max - R_min) / (num_geoms_total ÷ num_geoms_per_direction)
    # get pseudorandom direction vectors from Sobol sequence
    seq = skip(SobolSeq([-1, -1, -1], [1, 1, 1]), num_to_skip, exact=true)
    all_labels = Vector{String}[]
    all_geoms  = Matrix{Float64}[]
    for i in 1:(num_geoms_total ÷ num_geoms_per_direction)
        direction = Sobol.next!(seq)
        if norm(direction) == 0.0
            direction = Sobol.next!(seq)
        end
        normalize!(direction)
        R = rand(RotMatrix{3}) # get random orientation
        R_inner = R_min + (i-1) * radius_step_size
        R_outer = R_min + i * radius_step_size
        sampled_labels, sampled_geoms = get_random_dimer_geometries_along_direction_between_spheres(
            frag_1, labels_1, R * frag_2, labels_2, direction, R_inner, R_outer,
            num_geoms_per_direction
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
function sample_psuedorandom_dimers_in_spheres(
    fragment_1::AbstractVector{Matrix{Float64}}, labels_1::Vector{String},
    fragment_2::AbstractVector{Matrix{Float64}}, labels_2::Vector{String},
    R_min::Float64, R_max::Float64,
    num_geoms_total::Int=5000, num_geoms_per_direction::Int=5; num_to_skip::Int=0
)
    all_labels = Vector{String}[]
    all_geoms = Matrix{Float64}[]
    radius_step_size = (R_max - R_min) / (num_geoms_total ÷ num_geoms_per_direction)
    for i in 1:(num_geoms_total ÷ num_geoms_per_direction)
        frag_1 = fragment_1[rand(1:length(fragment_1))]
        frag_2 = fragment_2[rand(1:length(fragment_2))]
        R_inner = R_min + (i-1) * radius_step_size
        R_outer = R_min + i * radius_step_size
        sampled_labels, sampled_geoms = sample_psuedorandom_dimers_in_spheres(
            frag_1, labels_1, frag_2, labels_2,
            R_inner, R_outer,
            num_geoms_per_direction, num_geoms_per_direction,
            num_to_skip=(num_to_skip + i)
        )
        append!(all_labels, sampled_labels)
        append!(all_geoms, sampled_geoms)
    end
    return all_labels, all_geoms
end