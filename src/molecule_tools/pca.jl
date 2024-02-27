using LinearAlgebra, StatsBase, StaticArrays

function pca(
    coords::AbstractVector{MVector{3, Float64}},
    ref_coords::AbstractVector{Vector{MVector{3, Float64}}},
    h_gaussian::Float64,
    σ_gaussian::Float64,
    num_principal_components::Int=2
)
    @assert length(coords) == length(ref_coords[1]) "Coordinates and reference coordinates don't have the same size."
    n_dof = 3 * length(ref_coords[1])

    # allocate storage for flattened version of coordinates
    X = zeros(n_dof, length(ref_coords))
    for i in eachindex(ref_coords)
        @views X[:, i] = reduce(vcat, ref_coords[i])
    end

    centroid = sum(eachcol(X)) / n_dof
    for i in eachindex(ref_coords)
        @views X[:, i] -= centroid
    end
    C = X * X'
    evals, evecs = eigen(C)
    evals = evals[end-num_principal_components+1:end]
    P = evecs[:, end-num_principal_components+1:end]
    
    # The bias is simply the k largest eigenvalues added up and divided by σ^2
    e_bias = h_gaussian * exp(-sum(evals) / (2 * σ_gaussian^2))
    display(sum(evals))
    # HERE: Need to figure out why the eigenvalue is so large. Then test if I get the right gradient by finite difference.
    return e_bias

end