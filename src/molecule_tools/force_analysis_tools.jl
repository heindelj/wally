using LinearAlgebra
using Statistics
include("read_xyz.jl")

function dotsum3(frames::Array{Array{Float64,2},1})
    """
    Takes a series of matrices and sums the column-wise dot product along the series.
    """
    res = zeros(eltype(frames[begin]), size(frames[begin], 2))
    for i in LinearIndices(frames)
        for (j, col) in enumerate(eachcol(frames[i]))
            res[j] += dot(col, col)
        end
    end
    return res
end

function rms_force_on_each_atom(frames::Array{Array{Float64,2},1})
    """
    Takes a series of frames representing atomic forces on each atom and computes the rms
    forces on each atom. Returns a 1D array of the rms forces on each atom.
    """
    summed_squared_forces = dotsum3(frames)
    summed_squared_forces ./= length(frames)
    return sqrt.(summed_squared_forces)
end

function rms_force_by_atom_type(frames::Array{Array{Float64,2},1}, labels::AbstractArray)
    summed_squared_forces = dotsum3(frames)
    summed_squared_forces ./= length(frames)
    rms_force_per_atom = Dict{String, Float64}()
    number_of_each_atom = Dict{String, Int}()
    for (i, label) in enumerate(labels[1])
        if !(label in keys(rms_force_per_atom))
            rms_force_per_atom[label] = summed_squared_forces[i]
            number_of_each_atom[label] = 1
        else
            rms_force_per_atom[label] += summed_squared_forces[i]
            number_of_each_atom[label] += 1
        end
    end
    for key in keys(rms_force_per_atom)
        rms_force_per_atom[key] /= number_of_each_atom[key]
        rms_force_per_atom[key] = sqrt(rms_force_per_atom[key])
    end
    return rms_force_per_atom
end

function angles_between_vectors_in_two_matrices(A::AbstractMatrix, B::AbstractMatrix)
    """
    Takes two matrices, A and B, each of size mxN and computes the angle between all N m-dimensional vectors.
    Returns a vector containing these angles.
    """
    @assert axes(A) == axes(B)
    angle_vector = fill(0.0, size(A, 2))
    if !(A === B)
        for i in 1:size(A, 2)
            angle_vector[i] = acosd(dot(view(A, :, i), view(B, :, i)) / (norm(view(A, :, i)) * norm(view(B, :, i))))
        end
    end
    return angle_vector
end

function angle_vector_from_matrix_time_series(coords_A::Array{Array{Float64,2},1}, coords_B::Array{Array{Float64,2},1})
    """
    Takes two time series of 3xN matrices and computes the angle between each vector for each frame.
    Returns angles as a time series of vectors.
    """
    @assert axes.(coords_A) == axes.(coords_B)
    angle_vecs = Array{Array{Float64, 1}, 1}()
    for i_frame in 1:length(coords_A)
        push!(angle_vecs, angles_between_vectors_in_two_matrices(coords_A[i_frame], coords_B[i_frame]))
    end
    return angle_vecs
end

function average_angle_vector_by_atom_type(angle_vectors::Array{Array{Float64, 1}, 1})
    angles = fill(0.0, size(angle_vectors[begin]))
    collapse_angle_vectors = transpose(hcat(angle_vectors...))
    println(size(collapse_angle_vectors))
    for (i, vec) in enumerate(eachcol(collapse_angle_vectors))
        angles[i] = mean(vec)
    end
    return angles
end

@time header, labels, one_body_forces = read_xyz("data/1body_forces.dat")
@time header, labels, two_body_forces = read_xyz("data/2body_forces.dat")
@time angles = angle_vector_from_matrix_time_series(one_body_forces, two_body_forces)
@time display(average_angle_vector_by_atom_type(angles))