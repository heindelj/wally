using FiniteDifferences
using LinearAlgebra
include("atomic_masses.jl")
include("read_xyz.jl")
include("inertia_tensor.jl")

function get_second_derivative_on_stencil(f::Function, x_initial::AbstractVector, index::Int, dx::Float64, grid_size::Int, shape::Union{Tuple{Int}, Tuple{Int, Int}})
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
        @inbounds function_values[i] = f(reshape(geom, shape) * conversion(:bohr, :angstrom))
    end
    return dot(function_values, wts)
end

function get_mixed_second_derivative_on_stencil(f::Function, x_initial::AbstractVector, index1::Int, index2::Int, dx::Float64, grid_size::Int, shape::Union{Tuple{Int}, Tuple{Int, Int}})
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
        function_values[i] = f(reshape(geom, shape) * conversion(:bohr, :angstrom))
    end
    return dot(reshape(function_values, (length(dx_grid), length(dx_grid)))' * wts, wts)
end

function generate_hessian(f::Function, x_initial::AbstractArray, dx::Float64, on_diag_grid::Int, off_diag_grid::Int, shape::Union{Tuple{Int}, Tuple{Int, Int}})
    unweighted_hessian = zeros(length(x_initial), length(x_initial))
    x = vec(x_initial)
    # get just the lower triangle
    for i_col in 1:(length(x_initial)-1)
        for i_row in (i_col+1):length(x_initial)
            @inbounds unweighted_hessian[i_row, i_col] = get_mixed_second_derivative_on_stencil(f, x, i_row, i_col, dx, off_diag_grid, shape)
        end
    end
    unweighted_hessian += unweighted_hessian'

    # get the on diagonal elements
    for i in 1:length(x_initial)
        @inbounds unweighted_hessian[i, i] = get_second_derivative_on_stencil(f, x, i, dx, on_diag_grid, shape)
    end
    return unweighted_hessian
end

function inverse_mass_matrix(atom_labels::Vector{String})
    return diagm(1 ./ sqrt.(atomic_masses(repeat(atom_labels, inner=3)) * conversion(:amu, :au_mass)))
end

function generate_mass_weighted_hessian(f::Function, x_initial::AbstractArray, atom_labels::Vector{String}, dx::Float64, on_diag_grid::Int, off_diag_grid::Int, shape::Union{Tuple{Int}, Tuple{Int, Int}})
    inv_sqrt_mass_matrix = inverse_mass_matrix(atom_labels)
    unweighted_hessian = generate_hessian(f, x_initial, dx, on_diag_grid, off_diag_grid, shape)
    return inv_sqrt_mass_matrix * unweighted_hessian * inv_sqrt_mass_matrix
end

function insert_zero_eigvals!(eigvals::Vector{Float64}, num_to_insert::Int=6)
    for (i, eig) in enumerate(eigvals)
        if eig >= 0.0
            for _ in 1:num_to_insert
                insert!(eigvals, i, 0.0)
            end
            break
        end
    end
end

function harmonic_analysis(potential_function::Function, coords::AbstractArray, atom_labels::Vector{String}, projected_frequencies::Bool=false; dx::Float64=0.001, on_diag_grid::Int=5, off_diag_grid::Int=3)
    """
    Performs a harmonic analysis, returning the energies of each vibrational mode
    in wavenumbers, the eigenvectors in angstroms, and the reduced_masses in atomic units.
    """
    
    weighted_hessian = generate_mass_weighted_hessian(potential_function, coords * conversion(:angstrom, :bohr), atom_labels, dx, on_diag_grid, off_diag_grid, size(coords))
    
    # now get N_vib vectors orthogonal to the translations and rotations from QR factorization
    # ROBUSTNESS: handle the case of an atom and linear molecule
    # See https://gaussian.com/vib/ for syntax and reference
    translation_rotation_generator = infinitesimal_translation_and_rotation_matrix(coords * conversion(:angstrom, :bohr), atomic_masses(atom_labels) * conversion(:amu, :au_mass))
    D, _ = qr(translation_rotation_generator)

    # transform to translating and rotating frame and get eigvals
    if projected_frequencies
        rotating_frame_projection = D[:, 7:end]
        projected_hessian = rotating_frame_projection' * weighted_hessian * rotating_frame_projection
        evals = eigvals(projected_hessian)
        insert_zero_eigvals!(evals, 6) # ROBUSTNESS: atom and linear molecule changes number of modes
    else
        evals = eigvals(weighted_hessian)
    end
    # get the cartesian normal modes
    L = eigvecs(D' * weighted_hessian * D)
    normal_modes = inverse_mass_matrix(atom_labels) * D * L

    # normalize each column by sqrt of sum of inverse squares
    normalization = [sqrt(1 / sum(normal_modes[:,i].^2)) for i in 1:size(normal_modes, 1)]
    reduced_masses = normalization.^2
    final_normal_modes = eachcol(normal_modes) ./ normalization .* conversion(:bohr, :angstrom)
    return sqrt.(complex.(evals)) * conversion(:hartree, :wavenumbers), final_normal_modes, reduced_masses
end

function step_along_normal_mode(coords::AbstractVecOrMat{T}, mode_vec::AbstractVector{T}, num_steps_each_way::Int=5, step_size=750) where T <: AbstractFloat
    """
    Steps coords along cartesian normal mode num_steps_each_way symmetrically around coords. 
    All units assumed to be angstroms.
    If a matrix is passed in, it is assumed that calling vec(coords) will align the elements properly.
    """
    @assert length(coords) == length(mode_vec) "mode_vec and coords don't have same number of elements."
    final_shape = size(coords)
    displaced_coords = [vec(coords) + i * mode_vec * step_size for i in -num_steps_each_way:num_steps_each_way]
    return reshape.(displaced_coords, (final_shape,))
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
    evals, _, _ = harmonic_analysis(potential_function, geoms[begin], labels[begin])
    output_file::String = string(basename(input_file), "_frequencies.dat")
    save_frequencies(output_file, harmonic_frequencies(evals))
end
