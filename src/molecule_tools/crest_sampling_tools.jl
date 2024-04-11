include("molecular_axes.jl")
include("water_tools.jl")
include("read_xyz.jl")
using Random, ProgressMeter

"""
Takes a geometry of water molecules and replaces the location of one of the water molecules
with an ion.
"""
function replace_random_water_with_ion(coords::Matrix{Float64}, labels::Vector{String}, ion_label::String)
    water_indices = Int[]
    for i in eachindex(labels)
        if labels[i] == "O" || labels[i] == "H"
            push!(water_indices, i)
        end
    end
    non_water_indices = setdiff(1:length(labels), water_indices)

    water_coords = get_array_of_waters(coords[:, water_indices], labels[water_indices])

    random_index = rand(1:length(water_coords))
    out_coords = zeros(3, length(labels) - 2) # subtract three for removing water then add one for ion
    i_water = 1
    num_indices_used = 0
    out_labels = ["" for _ in 1:(length(labels)-2)]
    for i in eachindex(water_coords)
        if i != random_index
            out_coords[:, (3*(i_water-1)+1):(3*i_water)] = water_coords[i]
            out_labels[(3*(i_water-1)+1):(3*i_water)] = ["O", "H", "H"]
            num_indices_used += 3
            i_water += 1
        else
            ion_position = centroid(water_coords[i])
            out_coords[:, 3*(length(water_coords)-1)+1] = ion_position
            out_labels[3*(length(water_coords)-1)+1] = ion_label
            num_indices_used += 1
        end
    end
    # now add all of the non-water stuff that came with the cluster into the arrays we send back
    out_labels[(num_indices_used+1):end] = labels[non_water_indices]
    @views out_coords[:, (num_indices_used+1):end] = coords[:, non_water_indices]
    return out_labels, out_coords
end

function set_up_crest_sampling(seed_coordinates::Matrix{Float64}, seed_labels::Vector{String}, ion_label::String, num_sampling_runs::Int)
    for i in 1:num_sampling_runs
        mkpath(string("sample_", i))
        ion_water_labels, ion_water_coords = replace_random_water_with_ion(seed_coordinates, seed_labels, ion_label)
        write_xyz(string("sample_", i, "/ion_water_config_guess_", i, ".xyz"), ion_water_labels, ion_water_coords)
    end
end

function set_up_crest_sampling(seed_coordinates::Vector{Matrix{Float64}}, seed_labels::Vector{Vector{String}}, ion_label::String, num_sampling_runs::Int)
    rand_indices = rand(1:length(seed_coordinates), num_sampling_runs)
    for i in 1:num_sampling_runs
        mkpath(string("sample_", i))
        ion_water_labels, ion_water_coords = replace_random_water_with_ion(seed_coordinates[rand_indices[i]], seed_labels[rand_indices[i]], ion_label)
        write_xyz(string("sample_", i, "/ion_water_config_guess_", i, ".xyz"), ion_water_labels, ion_water_coords)
    end
end

function run_crest_sampling(charge::Int, gfn::Int=2)
    all_files_and_folders = readdir()

    for folder in all_files_and_folders
        if isdir(folder) && contains(folder, "sample_")
            cd(folder)
            i_sample = split(folder, "_")[2]
            crest_command = `crest ion_water_config_guess_$i_sample.xyz --nci --chrg $charge --gfn $gfn --noreftopo`
            run(crest_command)
            cd("..")
        end
    end
end

function identify_and_write_unique_geoms(xyz_outfile::String, energy_threshold::Float64=0.025)
    all_files_and_folders = readdir()

    all_geoms = Matrix{Float64}[]
    all_labels = Vector{String}[]
    all_energies = Float64[]
    for folder in all_files_and_folders
        if isdir(folder) && contains(folder, "sample_")
            cd(folder)
            headers, labels, geoms = read_xyz("crest_conformers.xyz")
            energies = [parse(Float64, split(header)[2]) for header in headers]
            append!(all_geoms, geoms)
            append!(all_labels, labels)
            append!(all_energies, energies)
            cd("..")
        end
    end
    sorted_by_energies = sortperm(all_energies)
    all_geoms = all_geoms[sorted_by_energies]
    all_labels = all_labels[sorted_by_energies]
    all_energies = all_energies[sorted_by_energies] .* 627.51
    all_energies .-= all_energies[1]
    unique_geom_bitmap = zeros(Int, Int(floor(all_energies[end] / energy_threshold) + 1))

    indices_to_keep = Int[]
    for i in eachindex(all_energies)
        target_index = Int(floor(all_energies[i] / energy_threshold)) + 1
        if unique_geom_bitmap[target_index] == 0
            unique_geom_bitmap[target_index] = 1
            push!(indices_to_keep, i)
        end
    end

    write_xyz(xyz_outfile, [string(length(all_labels[indices_to_keep][i]), "\n", all_energies[indices_to_keep][i]) for i in eachindex(all_labels[indices_to_keep])], all_labels[indices_to_keep], all_geoms[indices_to_keep])
end

