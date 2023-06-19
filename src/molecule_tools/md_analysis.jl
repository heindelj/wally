using LinearAlgebra
include("atomic_masses.jl")

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