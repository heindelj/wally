using LinearAlgebra, Rotations
include("molecular_graph.jl")
include("molecular_axes.jl")

function distance_scan(coords::Matrix{Float64}, indices::Tuple{Int, Int}, Δr_left::Float64=-0.3, Δr_right::Float64=0.5, nsteps::Int=9; aux_indices::Union{Tuple{Vector{Int}, Vector{Int}}, Nothing}=nothing)
    """
    Scans along a bond determined by two indices into the coords
    array passed in. The scan moves the coordinate corresponding
    to each index symmetrically. If aux_indices is not nothing, then all
    indices in the first vector are moved with index 1 and all
    indices in the second vector are moved with index two.

    Returns a copy nsteps copies of coords where the elements at indices
    have been appropriately modified. This could be optimized if needed.
    """
    @views r_ij = coords[:, indices[1]] - coords[:, indices[2]]
    coords_out = [copy(coords) for _ in 1:nsteps]

    displacements = LinRange(Δr_left, Δr_right, nsteps)
    t = [(norm(r_ij) + 0.5 * Δx) / norm(r_ij) for Δx in displacements]
    for i in 1:length(displacements)
        @views coords_out[i][:, indices[2]] = coords_out[i][:, indices[1]] - t[i] * r_ij
        @views coords_out[i][:, indices[1]] = coords_out[i][:, indices[1]] + (t[i] - 1.0) * r_ij
        if aux_indices !== nothing
            for i_aux_1 in aux_indices[1]
                @views coords_out[i][:, i_aux_1] += (t[i] - 1.0) * r_ij
            end
            for i_aux_2 in aux_indices[2]
                @views coords_out[i][:, i_aux_2] += (coords_out[i][:, indices[2]] - coords_out[i][:, indices[1]] + t[i] * r_ij)
            end
        end
    end
    return coords_out
end

function distance_scan(coords::Matrix{Float64}, scan_indices::Tuple{Tuple{Int, Int}, Tuple{Vector{Int}, Vector{Int}}}, Δr_left::Float64=-0.4, Δr_right::Float64=0.6, nsteps::Int=21)
    return distance_scan(coords, scan_indices[1], Δr_left, Δr_right, nsteps, aux_indices=scan_indices[2])
end

function angle_scan(coords::Matrix{Float64}, indices::Tuple{Int, Int, Int}, Δθ_in::Float64=-30.0, Δθ_out::Float64=30.0, nsteps::Int=13; aux_indices::Union{Tuple{Vector{Int}, Vector{Int}}, Nothing}=nothing)
    """
    Scans along an angle specified by three indices. Angle to scan in and out
    should be given in degrees. In means smaller angle, out means larger angle.
    Angle changes move both atoms symmetrically.
    """
    @views r_ij = coords[:, indices[1]] - coords[:, indices[2]]
    @views r_kj = coords[:, indices[3]] - coords[:, indices[2]]
    coords_out = [copy(coords) for _ in 1:nsteps]
    n = cross(normalize(r_ij), normalize(r_kj)) # normal to plane of r_ij and r_kj
    ΔΘ = LinRange(Δθ_in, Δθ_out, nsteps)
    for (i, θ) in enumerate(ΔΘ)
        R_left = AngleAxis(θ / 2 * π / 180.0, n[1], n[2], n[3])
        R_right = AngleAxis(θ / 2 * π / 180.0, -n[1], -n[2], -n[3])
        r_ij_new = convert(Vector{Float64}, R_left * normalize(r_ij)) * norm(r_ij)
        r_kj_new = convert(Vector{Float64}, R_right * normalize(r_kj)) * norm(r_kj)
        @views coords_out[i][:, indices[1]] = coords[:, indices[2]] + r_ij_new
        @views coords_out[i][:, indices[3]] = coords[:, indices[2]] + r_kj_new
        if aux_indices !== nothing
            for i_aux_1 in aux_indices[1]
                @views r_i_aux_1_j = coords[:, i_aux_1] - coords[:, indices[2]]
                r_i_aux_1_j_new = convert(Vector{Float64}, R_left * normalize(r_i_aux_1_j))
                @views coords_out[i][:, i_aux_1] = coords[:, indices[2]] + r_i_aux_1_j_new * norm(r_i_aux_1_j)
            end
            for i_aux_2 in aux_indices[2]
                @views r_i_aux_2_j = coords[:, i_aux_2] - coords[:, indices[2]]
                r_i_aux_2_j_new = convert(Vector{Float64}, R_right * normalize(r_i_aux_2_j))
                @views coords_out[i][:, i_aux_2] = coords[:, indices[2]] + r_i_aux_2_j_new * norm(r_i_aux_2_j)
            end
        end
    end
    return coords_out
end

function angle_scan(coords::Matrix{Float64}, scan_indices::Tuple{Tuple{Int, Int, Int}, Tuple{Vector{Int}, Vector{Int}}}, Δθ_in::Float64=-30.0, Δθ_out::Float64=30.0, nsteps::Int=13)
    return angle_scan(coords, scan_indices[1], Δθ_in, Δθ_out, nsteps, aux_indices=scan_indices[2])
end

function dihedral_scan(coords::Matrix{Float64}, indices::NTuple{4, Int}, Δθ::Float64=180.0, nsteps::Int=19; aux_indices::Union{Tuple{Vector{Int}, Vector{Int}}, Nothing}=nothing)
    """
    Scans over a dihedral angle formed by four indices. Angle to scan
    should be given in degrees. Angle changes move both atoms symmetrically.
    """
    @views r_ij = coords[:, indices[1]] - coords[:, indices[2]]
    @views r_jk = coords[:, indices[2]] - coords[:, indices[3]]
    @views r_kl = coords[:, indices[3]] - coords[:, indices[4]]

    coords_out = [copy(coords) for _ in 1:nsteps]
    bond_axis = normalize(r_jk)
    ΔΘ_all = LinRange(0, Δθ, nsteps)
    for (i, θ) in enumerate(ΔΘ_all)
        R_left  = AngleAxis(θ / 2 * π / 180.0,  bond_axis[1], bond_axis[2], bond_axis[3])
        R_right = AngleAxis(θ / 2 * π / 180.0, -bond_axis[1], -bond_axis[2], -bond_axis[3])
        r_ij_new = convert(Vector{Float64}, R_left * normalize(r_ij))
        r_kl_new = convert(Vector{Float64}, R_right * normalize(r_kl))
        @views coords_out[i][:, indices[1]] = coords[:, indices[2]] + r_ij_new * norm(r_ij)
        @views coords_out[i][:, indices[4]] = coords[:, indices[3]] - r_kl_new * norm(r_kl)
        if aux_indices !== nothing
            for i_aux_1 in aux_indices[1]
                @views r_i_aux_1_j = coords[:, i_aux_1] - coords[:, indices[3]]
                r_i_aux_1_j_new = convert(Vector{Float64}, R_left * normalize(r_i_aux_1_j))
                @views coords_out[i][:, i_aux_1] = coords[:, indices[3]] + r_i_aux_1_j_new * norm(r_i_aux_1_j)
            end
            for i_aux_2 in aux_indices[2]
                @views r_i_aux_2_j = coords[:, i_aux_2] - coords[:, indices[2]]
                r_i_aux_2_j_new = convert(Vector{Float64}, R_right * normalize(r_i_aux_2_j))
                @views coords_out[i][:, i_aux_2] = coords[:, indices[2]] + r_i_aux_2_j_new * norm(r_i_aux_2_j)
            end
        end
    end
    return coords_out
end

function dihedral_scan(coords::Matrix{Float64}, scan_indices::Tuple{NTuple{4, Int}, Tuple{Vector{Int}, Vector{Int}}}, Δθ::Float64=180.0, nsteps::Int=19)
    return dihedral_scan(coords, scan_indices[1], Δθ, nsteps, aux_indices=scan_indices[2])
end

function rotate_around_plane_normal(coords::Matrix{Float64}, plane_indices::Tuple{Int, Int, Int}, Δθ::Float64=360.0, nsteps::Int=37)
    """
    Takes three indices and determines the normal to a plane then rotates the atoms forming this
    plane around the normal vector. Currently doesn't accept any auxiliary indices.
    Centers rotation on middle index of plane.
    """
    i = plane_indices[1]
    j = plane_indices[2]
    k = plane_indices[3]
    n = cross(coords[:,j] - coords[:,i], coords[:,j] - coords[:,k])
    coords_out = [copy(coords) for _ in 1:nsteps]
    ΔΘ_all = LinRange(0, Δθ, nsteps)
    for (i_geom, θ) in enumerate(ΔΘ_all)
        @views coords_out[i_geom][:, [i, j, k]] = rotate_coords_around_axis_by_angle(coords_out[i_geom][:, [i, j, k]], n, θ * π / 180.0, coords[:,j])
    end
    return coords_out
end

function expand_triangle(coords::Matrix{Float64}, triangle_indices::Tuple{Int, Int, Int}, Δr_in::Float64=-0.5, Δr_out::Float64=2.0, nsteps::Int=26; aux_indices::Union{Tuple{Vector{Int}, Vector{Int}}, Nothing}=nothing)
    """
    Takes three indices and scans along the edge lengths of the triangle formed by these by
    simultaneously varying the rji and rjk vector lengths. Note this is essentially a symmetric
    stretching coordinate where the j atom is fixed. For this reason, we only accept aux_indices
    for the i and k atoms respectively.
    """
    i = triangle_indices[1]
    j = triangle_indices[2]
    k = triangle_indices[3]

    e_ji = normalize(coords[:, j] - coords[:, i])
    e_jk = normalize(coords[:, j] - coords[:, k])

    coords_out = [copy(coords) for _ in 1:nsteps]
    Δr_all = LinRange(Δr_in, Δr_out, nsteps)
    for (i_geom, Δr) in enumerate(Δr_all)
        coords_out[i_geom][:, i] -= e_ji * Δr
        coords_out[i_geom][:, k] -= e_jk * Δr
        if aux_indices !== nothing
            for i_aux_1 in aux_indices[1]
                @views coords_out[i_geom][:, i_aux_1] -= e_ji * Δr
            end
            for i_aux_2 in aux_indices[2]
                @views coords_out[i_geom][:, i_aux_2] -= e_jk * Δr
            end
        end
    end
    return coords_out
end

function two_dimensional_scan(coords::Matrix{Float64}, scan_1::Function, scan_2::Function)
    return reduce(vcat, scan_2.(scan_1(coords)))
end

function find_bond_breaking_distance_scans(mg::MolecularGraph)
    """
    Finds bonds to scan over by testing if that bond
    produces two disjoint fragements when the edge connecting them
    is deleted. A vector of a tuple of indices and a tuple of vectors
    of indices are returned. The former give you the indices over which
    we are scanning. The latter give you the auxiliary indices which are
    for all atoms in each disjoint fragment. These can then be used to
    preserve the relative positions of all atoms not being scanned.
    """
    valid_scans = Tuple{Tuple{Int, Int}, Tuple{Vector{Int}, Vector{Int}}}[]

    for edge in edges(mg.G)
        g2 = copy(mg.G)
        rem_edge!(g2, edge)
        if !is_connected(g2)
            push!(valid_scans, (
                (edge.src, edge.dst),
                (setdiff(connected_components(g2)[1], [edge.src]), 
                setdiff(connected_components(g2)[2], [edge.dst]))
            ))
        end
    end
    return valid_scans
end

function find_angle_scans(mg::MolecularGraph)
    """
    Finds unique angles which create disconnected fragments.

    Something still wrong here??
    """
    valid_scans = Tuple{NTuple{3, Int}, Tuple{Vector{Int}, Vector{Int}}}[]

    for j in vertices(mg.G)
        if degree(mg.G, j) >= 2
            g2 = copy(mg.G)
            rem_vertex!(g2, j)
            if !is_connected(g2)
                n = neighbors(mg.G, j)
                components = connected_components(g2)
                for i in 1:(length(components)-1)
                    for k in (i+1):length(components)
                        push!(valid_scans, (
                            (n[i], j, n[k]),
                            (setdiff(components[i], [j, n[i], n[k]]),
                             setdiff(components[k], [j, n[i], n[k]]))
                        ))
                    end
                end
            end
        end
    end
    return valid_scans
end

function fixed_path_length_search(g::SimpleGraph, depth::Int)
    """
    Finds all paths of specified length. Could easily add option to
    exclude rings but for now we leave rings in.

    Can be made faster by doing the DFS only up to the specified depth
    which Graphs.jl doesn't seem to have an option for.

    Also, doesn't respect symmetry of the molecule which would probably
    be rather difficult to incorporate here. Does remove reverse paths.
    """
    paths = Vector{Int}[]

    for v in vertices(g)
        path_lengths = spfa_shortest_paths(g, v)
        for i in eachindex(path_lengths)
            if path_lengths[i] == depth
                new_paths = yen_k_shortest_paths(g, v, i, maxdist=depth).paths
                for new_path in new_paths
                    if new_path ∉ paths && reverse(new_path) ∉ paths
                        push!(paths, new_path)
                    end
                end
            end
        end
    end
    return paths
end

function find_dihedral_scans(mg::MolecularGraph)
    """
    Finds dihedral rotations which move groups of atoms.
    The groups are identified by deleting the central bond
    from a graph and checking that the molecule is separated 
    into two disconnected graphs. Only unique bond rotations
    are returned since many possible dihedrals generate the same
    rotation.
    """
    valid_scans = Tuple{NTuple{4, Int}, Tuple{Vector{Int}, Vector{Int}}}[]

    possible_dihedral_scans = fixed_path_length_search(mg.G, 3)

    for possible_scan in possible_dihedral_scans
        g2 = copy(mg.G)
        rem_edge!(g2, Edge(possible_scan[2], possible_scan[3]))
        if !is_connected(g2)
            # determine if we have already stored a rotation around this bond
            is_new_scan = true
            for valid_scan in valid_scans
                if possible_scan[2] == valid_scan[1][2] && possible_scan[3] == valid_scan[1][3]
                    is_new_scan = false
                    break
                end
            end
            if is_new_scan
                push!(valid_scans, (
                    (possible_scan[1], possible_scan[2],
                    possible_scan[3], possible_scan[4]),
                    (setdiff(connected_components(g2)[1], [possible_scan[1], possible_scan[2]]), 
                    setdiff(connected_components(g2)[2], [possible_scan[3], possible_scan[4]]))
                ))
            end
        end
    end
    return valid_scans
end