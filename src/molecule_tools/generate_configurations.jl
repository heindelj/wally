include("molecular_axes.jl")
using LinearAlgebra

function evenly_spaced_indices(first::Int, last::Int, num_elems::Int)
    return Int.(round.(LinRange(first, last, num_elems)))
end

function scale_along_coordinates(coords::Matrix{Float64}, scale::Float64, center::Bool=true)
    if center
        coords .-= centroid(coords)
    end
    coords *= scale
    return coords
end
