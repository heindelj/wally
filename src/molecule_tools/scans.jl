using LinearAlgebra, Rotations
include("molecular_graph.jl")

# add option for auxiliary indices which will be moved with
# indices 1 and 2 so that the rest of the molecule can be kept
# rigid with the two atoms being moved. Will get these indices
# from analysis of the bond graph presumably.
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

function angle_scan(coords::Matrix{Float64}, indices::Tuple{Int, Int, Int}, Δθ_in::Float64=-30.0, Δθ_out::Float64=30.0, nsteps::Int=13)
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
        r_ij_new = convert(Vector{Float64}, R_left * normalize(r_ij))
        r_kj_new = convert(Vector{Float64}, R_right * normalize(r_kj))
        @views coords_out[i][:, indices[1]] = coords[:, indices[2]] + r_ij_new
        @views coords_out[i][:, indices[3]] = coords[:, indices[2]] + r_kj_new
    end
    return coords_out
end

function dihedral_scan(coords::Matrix{Float64}, indices::Tuple{Int, Int, Int, Int}, Δθ::Float64=60.0, nsteps::Int=13)
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
        @views coords_out[i][:, indices[1]] = coords[:, indices[2]] + r_ij_new
        @views coords_out[i][:, indices[4]] = coords[:, indices[3]] - r_kl_new
    end
    return coords_out
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