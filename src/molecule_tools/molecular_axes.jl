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