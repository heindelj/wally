include("units.jl")
using SpecialFunctions

function gradient_of_erf_bsse_fit(r1::AbstractVector{T}, r2::AbstractVector{T}, A::T, B::T) where T <: AbstractFloat
    """
    Takes the positions of two atoms and the fitting parameters, A and B, for the error function. Returns the approximate BSSE-correction to the gradient.

    r1 and r2 should be provided in angstroms. Gradients are returned in hartree/bohr

    Note that the gradient at r1 and r2 differ by a sign, but we pass back both for convenience.
    """
    gradient_at_r1 = (r1-r2) / norm(r1-r2)^2 * (-2*A*B*exp(-B^2*norm(r1-r2)^2)) / sqrt(π) * conversion(:kcal, :hartree) * conversion(:angstrom, :bohr)

    return gradient_at_r1, -gradient_at_r1
end

function energy_of_erf_bsse_fit(r1::AbstractVector{T}, r2::AbstractVector{T}, A::T, B::T) where T <: AbstractFloat
    return A*(1 + erf(-B*norm(r1-r2))) * conversion(:kcal, :hartree)
end

function total_energy_of_erf_bsse_fit(coords::Matrix{T}, atom_labels::Vector{String}, basis_set::String) where T <: AbstractFloat
    """
    Same as total_gradient_of_erf_bsse_fit but for the energy only
    """
    bsse_energy_correction = 0.0
    for i in 1:length(atom_labels)
        for j in (i+1):length(atom_labels)
            # This loop is only correct if the approximate energy is zero
            # when both A and B parameters are 0.0. Otherwise, we will need to
            # check that we got back (0.0, 0.0) as the parameters and just skip
            # these iterations.
            @views bsse_energy_correction += energy_of_erf_bsse_fit(coords[:,i], coords[:,j], error_function_fit_parameters(atom_labels[i], atom_labels[j], basis_set)...)
        end
    end
    return bsse_energy_correction
end

function total_gradient_of_erf_bsse_fit(coords::Matrix{T}, atom_labels::Vector{String}, basis_set::String) where T <: AbstractFloat
    """
    Looks up the appropriate A and B parameters for each atom pair and basis set. Then calculates the approximate BSSE gradient contribution and returns the (3,N) matrix containing these gradients.
    """
    bsse_gradient_correction = zeros(3, length(atom_labels))
    for i in 1:length(atom_labels)
        for j in (i+1):length(atom_labels)
            # This loop is only correct if the approximate gradient is zero
            # when both A and B parameters are 0.0. Otherwise, we will need to
            # check that we got back (0.0, 0.0) as the parameters and just skip
            # these iterations.
            @views grad_at_i, grad_at_j = gradient_of_erf_bsse_fit(coords[:,i], coords[:,j], error_function_fit_parameters(atom_labels[i], atom_labels[j], basis_set)...)
            @views bsse_gradient_correction[:,i] += grad_at_i
            @views bsse_gradient_correction[:,j] += grad_at_j
        end
    end
    return bsse_gradient_correction
end

function error_function_fit_parameters(atom_label_1::String, atom_label_2::String, basis_set::String)
    fit_params = Dict(("O", "O", "aug-cc-pvdz") => (14.435, 0.4639),
                      ("O", "O", "aug-cc-pvtz") => (9.445 , 0.4887),
                      ("O", "O", "aug-cc-pvqz") => (5.549 , 0.4998)
                     )
    # if we just don't have this pair of atoms and basis set, then return zeros for the correction.
    if !((uppercase(atom_label_1), uppercase(atom_label_2), lowercase(basis_set)) in keys(fit_params)) && !((uppercase(atom_label_2), uppercase(atom_label_1), lowercase(basis_set)) in keys(fit_params))
        return (0.0, 0.0)
    end
    # ensure we always read from the parameters in alphabetical order
    if (uppercase(atom_label_1) < uppercase(atom_label_2)) || (uppercase(atom_label_1) == uppercase(atom_label_2))
        return fit_params[(uppercase(atom_label_1), uppercase(atom_label_2), lowercase(basis_set))]
    end
    return fit_params[(uppercase(atom_label_2), uppercase(atom_label_1), lowercase(basis_set))]
end
