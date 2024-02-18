using StaticArrays, LinearAlgebra
include("vdw_radii.jl")

"""
Uses a fibonacci sequence to generate spiraling points which are successively
projected onto the surface of a sphere. This generates fairly evenly spaced
points on a sphere.

Radius controls the radius of the sphere the points will lie on.
n_points is number of points on the sphere.

See: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
"""
function fibonacci_sphere(radius::Float64, n_points::Int)

    points = SVector{3, Float64}[]
    ϕ = π * (sqrt(5.0) - 1.0)  # golden angle in radians

    for i in 0:(n_points-1)
        y = 1 - (i / (n_points-1)) * 2.0  # y goes from 1 to -1
        r = sqrt(1 - y * y)  # radius at y

        θ = ϕ * i  # golden angle increment

        x = cos(θ) * r
        z = sin(θ) * r

        push!(points, SVector{3, Float64}([x, y, z] * radius))
    end

    return points
end

"""
Computes the solvent accessible surface area of all provided atoms.
We compute this using the skrake rupley algorithm. We use the  
"""
function solvent_accessible_surface_area(
    coords::Matrix{Float64},
    labels::Vector{String},
    probe_radius::Float64=1.4,
    n_sphere_points::Int=960,
    custom_radii::Union{Nothing, Dict{String, Float64}}=nothing
)
    # convert coordinates to static array
    @views coords_static = [SVector{3, Float64}(coords[:, i]) for i in eachindex(eachcol(coords))]
    all_radii = zeros(length(labels))
    for i in eachindex(labels)
        if custom_radii !== nothing && haskey(custom_radii, labels[i])
            all_radii[i] = custom_radii[labels[i]] + probe_radius
        else
            all_radii[i] = vdw_radius(labels[i]) + probe_radius
        end
    end

    meshes = [fibonacci_sphere(1.0, n_sphere_points) for _ in 1:Threads.nthreads()]

    areas = zeros(length(labels))
    Threads.@threads for i in eachindex(coords_static)
        id = Threads.threadid()
        # move the spherical mesh to the radius of this atom centered on the atom
        for i_mesh in eachindex(meshes[id])
            meshes[id][i_mesh] *= all_radii[i]
            meshes[id][i_mesh] += coords_static[i]
        end

        r_i = all_radii[i] # distance of all test points from atom i
        num_accessible_points = 0
        for i_mesh in eachindex(meshes[id])
            is_accessible = true
            for j in eachindex(coords_static)
                if i != j
                    r_j = norm(meshes[id][i_mesh] - coords_static[j]) 
                    if r_j < r_i
                        is_accessible = false
                        break
                    end
                end
            end
            if is_accessible
                num_accessible_points += 1
            end
        end
        areas[i] = num_accessible_points / n_sphere_points * 4 * π * r_i^2

        # reset the mesh to a unit mesh
        for i_mesh in eachindex(meshes[id])
            meshes[id][i_mesh] -= coords_static[i]
            meshes[id][i_mesh] /= all_radii[i]
        end
    end
    return areas
end