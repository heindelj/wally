include("units.jl")
include("molecular_axes.jl")
using LinearAlgebra

function mass_weight_coordinates(x::AbstractArray,  masses::AbstractVector{<:AbstractFloat})
    shape = size(x)
    sqrt_mass_vec = 1 .* sqrt.(repeat(masses, inner=3))
    return reshape(sqrt_mass_vec .* vec(x), shape)
end

function inertia_tensor(geom::AbstractArray{<:AbstractFloat}, masses::AbstractVector{<:AbstractFloat})
    """
    Computes the moment of inertia tensor for a given geometry (in bohr) and masses (in atomic units).
    """
    __geom = reshape(copy(geom), (3, div(length(geom), 3)))
    __geom = __geom .- center_of_mass(__geom, masses)
    __geom = mass_weight_coordinates(geom, masses)

    return inertia_tensor(__geom)
end

function inertia_tensor(geom::AbstractArray{<:AbstractFloat})
    """
    Computes the moment of inertia tensor for a given mass-weighted geometry (in atomic units).
    """
    __geom = reshape(copy(geom), (3, div(length(geom), 3)))

    I = zeros(3, 3)
    x = 1
    y = 2
    z = 3
    # for each term see: https://scipython.com/book/chapter-6-numpy/problems/p65/the-moment-of-inertia-tensor/
    # Ixy, Ixz, Iyz 
    # PROBABLY SLOW
    I[y, x] = -dot(__geom[x,:], __geom[y,:])
    I[z, x] = -dot(__geom[x,:], __geom[z,:])
    I[z, y] = -dot(__geom[y,:], __geom[z,:])
    
    # get upper triangle as well
    I += I'

    # Ixx, Iyy, Izz
    I[x, x] = dot(__geom[y,:], __geom[y,:]) + dot(__geom[z,:], __geom[z,:])
    I[y, y] = dot(__geom[x,:], __geom[x,:]) + dot(__geom[z,:], __geom[z,:])
    I[z, z] = dot(__geom[x,:], __geom[x,:]) + dot(__geom[y,:], __geom[y,:])

    return I
end

function get_rotational_constants(I::AbstractMatrix)
    """
    Returns the rotational constants A, B, and C in wavenumbers given the inertia tensor.
    """
    principle_moments = eigvals(I)
    rotational_constants = 1 ./ ( 2 * (principle_moments)) * conversion(:hartree, :wavenumbers)
    return rotational_constants
end

function levi_civita_tensor_3()
    element = (i,j,k)->(i-j)*(j-k)*(k-i)/2
    ϵ = zeros(3,3,3)
    for i in 1:3
        for j in 1:3
            for k in 1:3
                ϵ[i,j,k] = element(i,j,k)
            end
        end
    end
    return ϵ
end

function infinitesimal_translation_and_rotation_matrix(geom::Array{<:AbstractFloat, 2}, masses::AbstractVector{<:AbstractFloat})
    """
    We build the matrix to transform the hessian to internal coordinates, which effectively
    projects out the translational and rotational DOFs from the Hessian.
    See: https://web.archive.org/web/20191229092611/https://gaussian.com/vib/
    """
    D = zeros(length(geom), 6)

    # shift to center of mass and mass-weight coordinates
    geom = geom .- center_of_mass(geom, masses)
    geom = mass_weight_coordinates(geom, masses)

    # generate translation vectors for mass-weighted cartesian coordinates
    for i in 1:length(masses)
        D[3*(i-1) + 1, 1] = sqrt(masses[i])
        D[3*(i-1) + 2, 2] = sqrt(masses[i])
        D[3*(i-1) + 3, 3] = sqrt(masses[i])
    end

    # get the infinitesimal rotation vectors
    R = zeros(length(geom), 3)
    I = inertia_tensor(geom)

    X = eigvecs(I)
    X_moments = eigvals(I)
    rotation_axes = X * diagm(1 ./ sqrt.(X_moments)) * X'

    ϵ = levi_civita_tensor_3()

    num_coords = length(geom)
    x_indices = 1:3:num_coords
    y_indices = 2:3:num_coords
    z_indices = 3:3:num_coords
    for n in 1:num_coords
        i = ((n - 1) ÷ 3) + 1
        if n in x_indices
            γ = 1
        elseif n in y_indices
            γ = 2
        elseif n in z_indices
            γ = 3
        end

        for k in 1:3
            for α in 1:3
                for β in 1:3
                    R[n, k] += rotation_axes[k, α] * ϵ[α, β, γ] * geom[β, i]
                end
            end
        end
    end
    R ./ norm.(eachcol(R))'

    for i in 1:3
        D[:,3+i] = R[:,i]
    end

    D = D ./ norm.(eachcol(D))'
    return D
end
