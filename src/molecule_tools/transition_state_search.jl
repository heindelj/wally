include("harmonic_frequencies.jl")

# Everything here is based on the Peng and Schlegel paper: http://chem.wayne.edu/schlegel/Pub_folder/162.pdf
# All of this assumes that the atoms come in sorted against one another.
# This is a standard requirement, but one can probably guess the correct ordering
# most of the time by finding the permutation which maximizes overlap of the 
# reactant and product vectors. Try this sometime for robustness.

# TODO: Find simple test system
# get gradients to be stored for potential
# implement the algorithm described in the paper

function midpoints_TS_guess(R::AbstractVector{T}, P::AbstractVector{T}) where T <: Real
    """
    Takes the midpoint of R and P as a naive TS guess geometry.
    """
    return 0.5 * (R + P)
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

