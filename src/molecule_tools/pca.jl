using LinearAlgebra, StatsBase, StaticArrays, Test

function pca(
    coords::AbstractVector{MVector{3, Float64}},
    ref_coords::AbstractVector{Vector{MVector{3, Float64}}},
    h_gaussian::Float64,
    σ_gaussian::Float64,
    num_principal_components::Int=2
)
    @assert length(coords) == length(ref_coords[1]) "Coordinates and reference coordinates don't have the same size."
    n_dof = 3 * length(ref_coords[1])

    coords_flat = reduce(vcat, coords)

    # allocate storage for flattened version of coordinates
    X = zeros(n_dof, length(ref_coords))
    for i in eachindex(ref_coords)
        @views X[:, i] = reduce(vcat, ref_coords[i])
    end

    centroid = sum(eachcol(X)) / n_dof
    for i in eachindex(ref_coords)
        @views X[:, i] -= centroid
    end
    coords_flat -= centroid

    C = X * X'
    evals, evecs = eigen(C)
    evals = evals[end-num_principal_components+1:end]
    P = evecs[:, end-num_principal_components+1:end]
    
    # The bias is simply the k largest eigenvalues added up and divided by σ^2
    e_bias = h_gaussian * exp(-(P' * coords_flat) ⋅ (P' * coords_flat) / (2 * σ_gaussian^2))
    
    grads = -h_gaussian / (σ_gaussian^2) * exp(-(P' * coords_flat) ⋅ (P' * coords_flat) / (2 * σ_gaussian^2)) * P * P' * coords_flat

    return e_bias, grads
end

function finite_difference_pca(
    coords::AbstractVector{MVector{3, Float64}},
    ref_coords::AbstractVector{Vector{MVector{3, Float64}}},
    h_gaussian::Float64,
    σ_gaussian::Float64,
    num_principal_components::Int=2,
    step_size=1e-5
)

    _, grads = pca(coords, ref_coords, h_gaussian, σ_gaussian, num_principal_components)
    
    fd_grads = [@MVector zeros(3) for _ in eachindex(coords)]
    for i in eachindex(coords)
        for w in 1:3
            coords[i][w] += step_size
            f_plus_h, _ = pca(coords, ref_coords, h_gaussian, σ_gaussian, num_principal_components)
            coords[i][w] -= 2 * step_size

            f_minus_h, _ = pca(coords, ref_coords, h_gaussian, σ_gaussian, num_principal_components)
            coords[i][w] += step_size
            fd_grads[i][w] += (f_plus_h - f_minus_h) / (2 * step_size)
        end
    end
    @test norm(reduce(vcat, fd_grads) - grads) < 1e-10
end