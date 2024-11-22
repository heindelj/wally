using Combinatorics
include("read_xyz.jl")
include("many_body_expansion.jl")

function single_point_energy(coords::AbstractMatrix{Float64}, labels::AbstractVector{String})
    exe = "/home/heindelj/installs/tinker_exes/analyze"
    params = "/home/heindelj/installs/tinker/params/amoebabio18.prm"
    write_xyz_with_connectivity_and_atom_types("sp_tinker.xyz", labels, coords)
    tinker_output = read(`$exe sp_tinker.xyz $params E`, String)
    for line in split(tinker_output, "\n")
        if occursin("Total Potential Energy", line)
            return parse(Float64, split(line)[5])
        end
    end
    @warn "Never found the total potential energy. Calculation likely failed. Returning 0.0."
    return 0.0
end

function mbe(coords::AbstractMatrix{Float64}, labels::AbstractVector{String}, fragment_indices::Vector{Vector{Int}}, max_order::Int)
    all_energies = [Float64[] for _ in 1:max_order]
    for i_order in 1:max_order
        all_subsystem_indices = combinations(fragment_indices, i_order)
        energies = zeros(length(all_subsystem_indices))
        for (i, subsystem_indices) in enumerate(all_subsystem_indices)
            flat_indices = reduce(vcat, subsystem_indices)
            energies[i] = single_point_energy(coords[:, flat_indices], labels[flat_indices])
        end
        all_energies[i_order] = energies
    end
    return get_mbe_data_from_subsystem_sums(sum.(all_energies), length(fragment_indices))
end