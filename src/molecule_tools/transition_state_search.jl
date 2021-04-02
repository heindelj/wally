include("harmonic_frequencies.jl")

# Everything here is based on the Peng and Schlegel paper: http://chem.wayne.edu/schlegel/Pub_folder/162.pdf
# All of this assumes that the atoms come in sorted against one another.
# This is a standard requirement, but one can probably guess the correct ordering
# most of the time by finding the permutation which maximizes overlap of the 
# reactant and product vectors. Try this sometime for robustness.

# TODO: implement the algorithm described in the paper

function midpoint_TS_guess(R::AbstractArray{T}, P::AbstractArray{T}) where T <: Real
    """
    Takes the midpoint of R and P as a naive TS guess geometry.
    """
    return 0.5 * (R + P)
end

function linear_midpoint(R::AbstractArray{T}, P::AbstractArray{T}) where T <: Real
    return 0.5 * (R + P)
end

function spherical_midpoint(R::AbstractArray{T}, P::AbstractArray{T}) where T <: Real
    """
    Calculates the midpoint of each vector in R and P by bisecting the point
    as if it were on a sphere. I then interpolate the point to halfway between
    the norms of the two endpoints.
    """
    reshape_on_return = false
    if length(size(R)) == 1
        reshape_on_return = true
        R = reshape(R, (3, :))
        P = reshape(P, (3, :))
    end

    midpoint = zero(R)

    for i in 1:size(R)[2]
        midpoint_temp = 0.5 * (R[:,i] / norm(R[:,i]) + P[:,i] / norm(P[:,i]))
        midpoint[:,i] = midpoint_temp * (0.5 * (norm(R[:,i]) + norm(P[:,i])))
    end

    if reshape_on_return
        return vec(midpoint)
    end
    return midpoint
end

function tangent_vector(R::AbstractVector{T}, P::AbstractVector{T}) where T <: Real
    """
    Returns the tangent vector used in linear synchronous transit.
    """
    return (P - R) / norm(P - R)
end

function tangent_vector(R::AbstractVector{T}, P::AbstractVector{T}, X::AbstractVector{T}) where T <: Real
    """
    Returns the tangent vector used in quadratic synchronous transit.
    See: http://chem.wayne.edu/schlegel/Pub_folder/162.pdf
    """
    a_squared::T = (R - X) ⋅ (R - X) * (P - X) ⋅ (P - X) / 
    ((R - X) ⋅ (R - X) + (P - X) ⋅ (P - X) - 2 * (R - X) ⋅ (P - X))

    tangent = sqrt(a_squared) * ( (P - X) / ((P - X) ⋅ (P - X)) - (R - X) / ((R - X) ⋅ (R - X)) )
    return tangent
end

