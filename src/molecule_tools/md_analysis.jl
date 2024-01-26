using LinearAlgebra, ProgressBars
include("atomic_masses.jl")
include("molecular_cluster.jl")

function compute_density_droplet(
    trajectory::Vector{Matrix{Float64}},
    labels::Vector{Vector{String}},
    max_radius::Float64,
    n_bins::Int
)
    radii = LinRange(0.0, max_radius, n_bins+1)
    step_size = max_radius / (n_bins + 1)
    densities = zeros(n_bins)
    for i in eachindex(trajectory)
        for i_atom in eachindex(labels[i])
            dist = norm(trajectory[i][:, i_atom])
            bin_index = Int(floor(dist / step_size)) + 1
            if bin_index <= n_bins
                densities[bin_index] += label_to_mass(Symbol(labels[i][i_atom]))
            end
        end
    end
    for i in eachindex(densities)
        densities[i] /= length(trajectory) # get average mass in bin per frame
        densities[i] /= 6.022*10^23 # g/mol to g
        densities[i] /= ((4 / 3) * π * (radii[i+1]^3 - radii[i]^3))
        densities[i] *= 10^24 # A^-3 to cm^-3
    end
    return densities
end

function compute_density_slab(
    trajectory::Vector{Matrix{Float64}},
    labels::Vector{Vector{String}},
    L_x::Float64,
    L_y::Float64,
    L_z::Float64,
    n_bins::Int
)
    slices = LinRange(0.0, max_radius, n_bins+1)
    step_size = max_radius / (n_bins + 1)
    densities = zeros(n_bins)
    for i in eachindex(trajectory)
        for i_atom in eachindex(labels[i])
            dist = norm(trajectory[i][:, i_atom])
            bin_index = Int(floor(dist / step_size)) + 1
            if bin_index <= n_bins
                densities[bin_index] += label_to_mass(Symbol(labels[i][i_atom]))
            end
        end
    end
    for i in eachindex(densities)
        densities[i] /= length(trajectory) # get average mass in bin per frame
        densities[i] /= 6.022*10^23 # g/mol to g
        densities[i] /= ((4 / 3) * π * (radii[i+1]^3 - radii[i]^3))
        densities[i] *= 10^24 # A^-3 to cm^-3
    end
    return densities
end

function find_average_distance_between_molecular_centers(cluster::Cluster, molecule_labels::Vector{String}, max_distance::Float64=maxintfloat(Float64))
    center_indices = molecules_by_formula(cluster, molecule_labels)
    if length(center_indices) < 2
        return 0.0
    end
    total_distance = 0.0
    npairs = 0
    for i in 1:length(center_indices)-1
        for j in (i+1):length(center_indices)
            distance = norm(cluster.centers[i] - cluster.centers[j])
            if distance < max_distance
                npairs += 1
                total_distance += distance
            end
        end
    end

    return total_distance / npairs
end

"""
Computes distance of each point from reference point for each atom. The number
of reference points and coordinates must be the same.
"""
function compute_total_distance_travelled_from_reference_structure(trajectory::Vector{Vector{MVector{3, Float64}}}, ref_structure::Vector{MVector{3, Float64}})
    @assert length(trajectory[1]) == length(ref_structure)
    distances = [zeros(length(trajectory)) for _ in eachindex(ref_structure)]

    for i_frame in eachindex(trajectory)
        for i_coord in eachindex(trajectory[i_frame])
            distances[i_coord] = norm(trajectory[i_frame][i_coord] - ref_structure[i_coord])
        end
    end
    return distances
end



function get_trajectory_of_molecular_fragments(trajectory::Vector{Vector{MVector{3, Float64}}}, labels::Vector{Vector{String}}, molecule_labels::Vector{String})
    fragment_trajectory = Vector{MVector{3, Float64}}[]
    cluster = build_cluster(trajectory[1], labels[1])
    center_indices = molecules_by_formula(cluster, molecule_labels)
    #reference_coords = trajectory[center_indices]
    #Threads.@threads for i_frame in ProgressBar(eachindex(trajectory))
    #    cluster = build_cluster(trajectory[i_frame], labels[i_frame])
    #    center_indices = molecules_by_formula(cluster, molecule_labels)

    #TODO: FINISH WRITING THIS FUNCTION!!!!

end