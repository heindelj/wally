include("atomic_masses.jl")
using LinearAlgebra
using Rotations
using Test

function center_of_mass(geom::AbstractArray{<:AbstractFloat}, masses::AbstractVector{<:AbstractFloat})
    return sum(reshape(repeat(masses, inner=3) .* vec(geom) / sum(masses), (3, :)), dims=2)
end

function center_of_mass_distance(sub_structure::AbstractMatrix, sub_structure_masses::AbstractVector, full_structure::AbstractMatrix, full_structure_masses::AbstractVector)
    return norm(center_of_mass(sub_structure, sub_structure_masses) - center_of_mass(full_structure, full_structure_masses))
end

#############################################
### CARTESIAN TO INTERNALS AND BACK AGAIN ###
#############################################

function angle(r_12::AbstractVector, r_23::AbstractVector)
    return acosd((r_12 ⋅r_23)/(norm(r_12) * norm(r_23)))
end

function dihedral_angle(r_12::AbstractVector, r_23::AbstractVector, r_34::AbstractVector)
    """ See: https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates """
    r_12 /= norm(r_12)
    r_23 /= norm(r_23)
    r_34 /= norm(r_34)

    n1 = cross(r_12, r_23)
    n2 = cross(r_23, r_34)
    m1 = cross(n1, r_23)
    return atand(m1 ⋅n2, n1 ⋅n2)
end

function zmat_to_string(zmat::Vector{Vector{Tuple{Int, Float64}}}, labels::Vector{String})
    @assert length(labels) == length(zmat) "Number of atom labels and z-matrix entries are not equal."
    complete_string = ""
    for i in 1:length(zmat)
        complete_string = string(complete_string, labels[i])
        for j in 1:length(zmat[i])
            if zmat[i][j][1] != 0
                complete_string = string(complete_string, " ", zmat[i][j][1], " ", zmat[i][j][2])
            else
                continue
            end
        end
        complete_string = string(complete_string, "\n")
    end
    return complete_string
end

function xyz_to_zmat(coords::Matrix{<:AbstractFloat})
    """
    Converts a matrix of cartesian coordinates into a zmatrix.
    Each entry in the zmatrix is stored as a Tuple{Int, Float64} where the Int says which atom
    this entry is relative to, and the Float64 is the value of that entry (distance, angle, or dihedral).
    Every atom will have three entries, and the integer will be a zero if that entry should be ignored.
    That is, the first atom will have [(0, 0.0), (0, 0.0), (0, 0.0)], while the second atom
    will have a valid entry for the first element in this vector.
    The whole zmatrix is returned as a Vector{Vector{Tuple{Int, Float64}}}
    """
    zmat::Vector{Vector{Tuple{Int, Float64}}} = [[(0, 0.0) for _ in 1:3] for _ in 1:size(coords, 2)]
    for i_atom in 1:size(coords, 2)
        if i_atom == 1
            continue
        elseif i_atom == 2
            zmat[i_atom][1] = (i_atom-1, norm(coords[:, i_atom] - coords[:, i_atom-1]))
        elseif i_atom == 3
            zmat[i_atom][1] = (i_atom-1, norm(coords[:, i_atom] - coords[:, i_atom-1]))
            zmat[i_atom][2] = (i_atom-2, 180.0 - angle(coords[:, i_atom] - coords[:, i_atom-1], coords[:, i_atom-1] - coords[:, i_atom-2]))
        else
            # bond distance
            zmat[i_atom][1] = (i_atom-1, norm(coords[:, i_atom] - coords[:, i_atom-1]))

            # bond angle
            zmat[i_atom][2] = (i_atom-2, 180.0 - angle(coords[:, i_atom] - coords[:, i_atom-1], coords[:, i_atom-1] - coords[:, i_atom-2]))
            
            # dihedral angle
            zmat[i_atom][3] = (i_atom-3, dihedral_angle(coords[:, i_atom] - coords[:, i_atom-1], coords[:, i_atom-1] - coords[:, i_atom-2], coords[:, i_atom-2] - coords[:, i_atom-3]))   
        end
    end
    return zmat
end

### KABSCH CODE ADAPTED FROM BIOMOLECULARSTRUCTURES.jl https://github.com/hng/BiomolecularStructures.jl/blob/master/src/KABSCH/kabsch.jl ###


# Calculate root mean square deviation of two matrices A, B
# http://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions
function rmsd(A::AbstractArray{T}, B::AbstractArray{T}) where T <: Real

    RMSD::Float64 = 0.0

    # N coordinates
    N::Int = length(A)

    for i in 1:N
        RMSD += (A[i] - B[i])^2
    end
    return sqrt(RMSD / N / 3.0)
end

# calculate a centroid of a matrix
function centroid(m::AbstractArray{T,2}) where T <: Real
    return sum(m, dims=2) ./ size(m)[2]
end

# Translate P, Q so centroids are the origin of the coordinate system (row-wise P and Q)
function translate_points(P::AbstractArray{T,2}, Q::AbstractArray{T,2}) where T <: Real
    return P .- centroid(P')', Q .- centroid(Q')'
end

# Input: Two sets of points: ref_coords, coords as 3xN Matrices
# returns optimally rotated matrix 
function kabsch(ref_coords::AbstractArray{T,2}, coords::AbstractArray{T,2}) where T <: Real

    @assert size(ref_coords) == size(coords) "Reference structure and actual don't the have same shape."
    @assert size(ref_coords)[1] == 3 "Coordinates should be 3xN matrices."

    # Everything happens on row-ordered matrices.
    # @speed change operations to be on column-ordered matrices.
    ref_coords = ref_coords'
    coords = coords'

    centered_reference::Array{T,2}, centered_coords::Array{T,2} = translate_points(ref_coords, coords)
    # Compute covariance matrix A
    A::Array{T,2} = *(centered_coords', centered_reference)

    # Calculate Singular Value Decomposition (SVD) of A
    u::Array{T,2}, d::Array{T,1}, vt::Array{T,2} = svd(A)

    # check for reflection
    f::Int64 = sign(det(vt) * det(u))
    m::Array{Int64,2} = [1 0 0; 0 1 0; 0 0 f]

    # Calculate the optimal rotation matrix _and_ superimpose it
    return broadcast(+, *(centered_coords, u, m, vt'), centroid(ref_coords')')'

end

function kabsch(ref_coords::AbstractVector{T}, coords::AbstractVector{T}) where T <: Real
    @assert isinteger(length(ref_coords) / 3) "Reference structure length isn't divisible by 3. Are these cartesian coordinates"
    @assert isinteger(length(coords) / 3) "Actual structure length isn't divisible by 3. Are these cartesian coordinates"
    ref_coords = reshape(ref_coords, (3, :))
    coords = reshape(coords, (3, :))
    return vec(kabsch(ref_coords, coords))
end

# directly return RMSD for matrices P, Q for convenience
function kabsch_rmsd(P::AbstractArray{T}, Q::AbstractArray{T}) where T <: Real
    return rmsd(P,kabsch(P,Q))
end

A = [[ 0.84305,   1.79893,    0.521059] [ 1.61653,   1.24596,    0.316728] [ 1.17692,   2.66457,    0.740568] [-0.730885,  0.553893,  -1.56561 ]]
R = rand(RotMatrix{3})
@test kabsch(A, R * A) ≈ A
