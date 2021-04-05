using FiniteDifferences
using LinearAlgebra
include("atomic_masses.jl")
include("read_xyz.jl")
include("inertia_tensor.jl")

function get_second_derivative_on_stencil(f::Function, x_initial::AbstractVector, index::Int, dx::Float64, grid_size::Int)
    """
    Gets the on-diagonal second derivatives from an arbitrary order finite difference.
    Recommned grid size for Hessians is 5 (i.e. 5-poiny stencil).
    """
    stencil_coordinates = reshape(repeat(x_initial, grid_size), (length(x_initial), grid_size))
    
    fdm5_2 = central_fdm(grid_size, 2)
    dx_grid = fdm5_2.grid * dx
    wts = fdm5_2.coefs / dx^2
    
    for (i, displacement) in enumerate(dx_grid)
        stencil_coordinates[index, i] += displacement
    end
    function_values = zeros(length(dx_grid))
    for (i, geom) in enumerate(eachcol(stencil_coordinates))
        @inbounds function_values[i] = f(geom * conversion(:bohr, :angstrom))
    end
    return dot(function_values, wts)
end

function get_mixed_second_derivative_on_stencil(f::Function, x_initial::AbstractVector, index1::Int, index2::Int, dx::Float64, grid_size::Int)
    """
    Using 3x3 point 2-D stencil for mixed partial derivatives.
    """
    stencil_coordinates = reshape(repeat(x_initial, grid_size^2), (length(x_initial), grid_size^2))
    fdm3_1 = central_fdm(grid_size, 1)
    dx_grid = fdm3_1.grid * dx
    wts = fdm3_1.coefs / dx
    for (i, displacement_1) in enumerate(dx_grid)
        for (j, displacement_2) in enumerate(dx_grid)
            @inbounds stencil_coordinates[index1, (i-1)*length(dx_grid) + j] += displacement_1
            @inbounds stencil_coordinates[index2, (i-1)*length(dx_grid) + j] += displacement_2
        end
    end
    function_values = zeros(length(dx_grid)^2)
    for (i, geom) in enumerate(eachcol(stencil_coordinates))
        function_values[i] = f(geom * conversion(:bohr, :angstrom))
    end
    return dot(reshape(function_values, (length(dx_grid), length(dx_grid)))' * wts, wts)
end

function generate_hessian(f::Function, x_initial::AbstractArray, dx::Float64, on_diag_grid::Int, off_diag_grid::Int)
    unweighted_hessian = zeros(length(x_initial), length(x_initial))
    x = vec(x_initial)
    # get just the lower triangle
    for i_col in 1:(length(x_initial)-1)
        for i_row in (i_col+1):length(x_initial)
            @inbounds unweighted_hessian[i_row, i_col] = get_mixed_second_derivative_on_stencil(f, x, i_row, i_col, dx, off_diag_grid)
        end
    end
    unweighted_hessian += unweighted_hessian'

    # get the on diagonal elements
    for i in 1:length(x_initial)
        @inbounds unweighted_hessian[i, i] = get_second_derivative_on_stencil(f, x, i, dx, on_diag_grid)
    end
    return unweighted_hessian
end

function generate_mass_weighted_hessian(f::Function, x_initial::AbstractArray, atom_labels::Vector{String}, dx::Float64, on_diag_grid::Int, off_diag_grid::Int)
    inv_sqrt_mass_matrix = diagm(1 ./ sqrt.(atomic_masses(repeat(atom_labels, inner=3)) * conversion(:amu, :au)))
    unweighted_hessian = generate_hessian(f, x_initial, dx, on_diag_grid, off_diag_grid)
    return inv_sqrt_mass_matrix * unweighted_hessian * inv_sqrt_mass_matrix
end

function harmonic_frequencies(potential_function::Function, coords::AbstractArray, atom_labels::Vector{String}; dx::Float64=0.001, on_diag_grid::Int=5, off_diag_grid::Int=3, return_eigvecs::Bool=false, return_eigvals::Bool=false)
    """
    Computes the harmonic frequencies of a provided potential function.
    If return eigvals is true, the eigenvalues of the hessian (rather than the frequencies)
    will be returned in atomic units. Frequencies are returned in wavenumbers.
    Note that the coords will be passed to the function as a vector with units of angstroms.
    """
    weighted_hessian = generate_mass_weighted_hessian(potential_function, coords * conversion(:angstrom, :bohr), atom_labels, dx, on_diag_grid, off_diag_grid)
    if return_eigvecs
        normal_modes = eigvecs(weighted_hessian)
        if !return_eigvals
            frequencies = sqrt.(complex.(eigvals(weighted_hessian)) ) * conversion(:hartree, :wavenumbers)
            return normal_modes, frequencies
        else
            return normal_modes, eigvals(weighted_hessian)
        end
    else
        if !return_eigvals
            return sqrt.(complex.(eigvals(weighted_hessian)) ) * conversion(:hartree, :wavenumbers)
        else
            return eigvals(weighted_hessian)
        end
    end
end

function projected_harmonic_frequencies(potential_function::Function, coords::AbstractArray, atom_labels::Vector{String}; dx::Float64=0.001, on_diag_grid::Int=5, off_diag_grid::Int=3, return_eigvecs::Bool=false, return_eigvals::Bool=false)
    weighted_hessian = generate_mass_weighted_hessian(potential_function, coords * conversion(:angstrom, :bohr), atom_labels, dx, on_diag_grid, off_diag_grid)
    
    # now get N_vib vectors orthogonal to the translations and rotations from QR factorization
    # ROBUSTNESS: handle the case of an atom and linear molecule
    translation_rotation_generator = infinitesimal_translation_and_rotation_matrix(coords * conversion(:angstrom, :bohr), atomic_masses(atom_labels) * conversion(:amu, :au))
    Q, R = qr(translation_rotation_generator)
    mw_projection_matrix = @view(Q[:, 7:end])

    # transform to translating and rotating frame
    projected_hessian = mw_projection_matrix' * weighted_hessian * mw_projection_matrix

    if return_eigvecs
        normal_modes = eigvecs(projected_hessian)
        if !return_eigvals
            frequencies = sqrt.(complex.(eigvals(projected_hessian)) ) * conversion(:hartree, :wavenumbers)
            return normal_modes, frequencies
        else
            return normal_modes, eigvals(projected_hessian)
        end
    else
        if !return_eigvals
            return sqrt.(complex.(eigvals(projected_hessian)) ) * conversion(:hartree, :wavenumbers)
        else
            return eigvals(projected_hessian)
        end
    end
end

function step_along_vector(coords::AbstractVecOrMat{T}, mode_vec::AbstractVector{T}, step_size::Float64=0.5) where T <: AbstractFloat
    """
    Steps coords along mode_vec by a distance step_size. All units assumed to be angstroms.
    If a matrix is passed on, it is assumed that calling vec(coords) will align the elements properly.
    """
    @assert length(coords) == length(mode_vec) "mode_vec and coords don't have same number of elements."
    final_shape = size(coords)
    displaced_coords = vec(coords)
    return reshape(displaced_coords + step_size * mode_vec, final_shape)
end

function get_zpe(frequencies::Array{Complex{T},1}) where T<:AbstractFloat
    """
    Computes harmonic zero point energy from frequencies calculated above. The frequencies will be complex numbers
    to show the possible imaginary frequencies. The lowest six eigenvalues will just be ignored.
    Might want to add a cutoff to the frequencies so this will work for transition states.
    """
    zpe::Float64 = 0.0
    for freq in frequencies[7:end]
        zpe += Real(freq)
    end
    return 0.5 * sum(zpe)
end

function save_frequencies(file_name::AbstractString, frequencies::Array{Complex{T},1}) where T<:AbstractFloat
    """
    File to save frequencies in a nicely formatted way
    """
    open(file_name, "w") do io
        for i in eachindex(frequencies)
            write(io, string(i, ": ", frequencies[i]), "\n")
        end
        write(io, "zpe: ", string(get_zpe(frequencies)))
    end
end

function get_frequencies_from_xyz(input_file::AbstractString, potential_function::Function)
    _, labels, geoms = read_xyz(input_file)
    frequencies = harmonic_frequencies(potential_function, geoms[begin], labels[begin])
    output_file::String = string(basename(input_file), "_frequencies.dat")
    save_frequencies(output_file, frequencies)
end