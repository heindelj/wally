include("harmonic_frequencies.jl")

# Everything here is based on the Peng and Schlegel paper: http://chem.wayne.edu/schlegel/Pub_folder/162.pdf
# All of this assumes that the atoms come in sorted against one another.
# This is a standard requirement, but one can probably guess the correct ordering
# most of the time by finding the permutation which maximizes overlap of the 
# reactant and product vectors. Try this sometime for robustness.

# TODO: implement the algorithm described in the paper

function rms(grad::AbstractVector)
    RMSD::Float64 = 0.0

    for i in 1:length(grad)
        RMSD += grad[i]^2
    end
    return sqrt(RMSD / length(grad))
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

function tangent_displacement_magnitude(tangent::AbstractVector{T}, gradient::AbstractVector{T}, evec::AbstractVector{T}, eigval::T) where T <: Real
    """
    Determine size of step along tangent from eq. (11) of http://chem.wayne.edu/schlegel/Pub_folder/162.pdf
    Takes everything in atomic units.
    """
    λ = (eigval + sqrt(eigval^2 + 4 * (evec ⋅ gradient)^2)) * 0.5
    Δx = -evec * (evec ⋅ gradient) / (eigval - λ)
    return dot(Δx / norm(Δx), tangent)
end

function transition_state_search(labels::Vector{String}, energy_function::Function, gradient_function::Function, R::AbstractVector, P::AbstractVector, X::Union{AbstractVecOrMat, Nothing}=nothing;
    num_steps::Int=100,
    tangent_overlap_cutoff::Float64=0.8,
    tangent_step_size_cutoff::Float64=0.05,
    step_size::Float64=0.1,
    max_step_size::Float64=0.3)
    """
    Does a transition state seacrh on the potential function trying to connect points R and P
    (reactant and product). The potential should take a geometry as a vector and return the energy.
    Optionally, a third point X can be supplied as a guess. If no guess is supplied, the midpoint
    of R and P will be used as the initial guess. After this, we will always use the quadratic
    synchronous transit tangent vector.
    Here is the algorithm:
    1. Update the Hessian
    2. Compute the tangent vector
    3. Determine whether to climb along the tangent or follow an eigenvector
        a. First two steps are always climbing steps
        b. Steps 3 and 4 can be either climbing or following based on a cutoff parameter
        c. Always choose eigenvector following after this point
    4. Move the molecule
        a. For a climbing step, compute the magnitude of the displacement from tangent_displacement_magnitude.
        b. For an evec-following step, follow the smallest eigenvalue (most imaginary frequency)
        or the one with a large overlap with a tangent vector (min_tangent_overlap=0.8). Step size along even is also
        a parameter (step size). 0.1 bohr is usually good.
    """
    visited_geometries::Array{Array{Float64, 2}, 1} = []
    
    # GET THE CURRENT POSITION
    if X === nothing
        X = linear_midpoint(R, P)
    end

    # PUT EVERYTHING IN BOHR
    R *= conversion(:angstrom, :bohr)
    P *= conversion(:angstrom, :bohr)
    X *= conversion(:angstrom, :bohr)

    push!(visited_geometries, reshape(X, (3,:)))

    # BEGIN LOOPING ALGORITHM
    for i in 1:num_steps
        # UPDATE HESSIAN, GRADIENTS, AND TANGENT VECTOR
        eigvecs, eigvals = harmonic_frequencies(energy_function, X * conversion(:bohr, :angstrom), labels, on_diag_grid=3, return_eigvecs=true, return_eigvals=true)
        tan_vector = tangent_vector(R, P, X)
        gradient = gradient_function(X * conversion(:bohr, :angstrom))
        
        if rms(gradient) < 0.0001
            println("Finished!")
            println("Current frequencies are:")
            display(sqrt.(complex.(eigvals) ) * conversion(:hartree, :wavenumbers))
            break
        end

        # CHOOSE WHICH EVEC TO FOLLOW
        evec_index = 1
        overlaps = dot.((tan_vector,), eachcol(evecs))
        if maximum(abs.(overlaps)) >= tangent_overlap_cutoff
            evec_index = argmax(abs.(overlaps))
            println("Switching to following eigenvector ", evec_index, " with frequency ", sqrt(complex(eigvals[evec_index])) * conversion(:hartree, :wavenumbers), " cm^-1")
        end

        # CHOOSE BETWEEN CLIMBING AND EIGVEC FOLLOWING
        if i <= 2
            step = tangent_displacement_magnitude(tan_vector, gradient, eigvecs[:,evec_index], eigvals[evec_index])
            step = min(max_step_size, step)
            X += step * X
            println("Step ", i, ": Climbing by ", step, " a.u.")
        elseif i == 3 || i == 4
            step = tangent_displacement_magnitude(tan_vector, gradient, eigvecs[:,evec_index], eigvals[evec_index])
            if step < tangent_step_size_cutoff
                X += step_size * eigvecs[:, evec_index]
                println("Step ", i, ": Following eigenvector ", evec_index)
            else
                step = tangent_displacement_magnitude(tan_vector, gradient, eigvecs[:,evec_index], eigvals[evec_index])
                step = min(max_step_size, step)
                X += step * X
                println("Step ", i, ": Climbing by ", step, " a.u.")
            end
        else
            X += step_size * eigvecs[:, evec_index]
            println("Step ", i, ": Following eigenvector ", evec_index)
        end
        println("Step ", i, ": RMS Gradient = ", rms(gradient), " a.u.")
        push!(visited_geometries, reshape(X, (3,:)))
    end
    return visited_geometries
end