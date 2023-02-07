using LinearAlgebra, CSV, DataFrames
include("read_xyz.jl")

"""
Parses the wiberg bond order matrix from a Q-Chem output file.
Returns an array of all bond order matrices found in the file.
"""
function parse_wiberg_BO_matrix(output_file::String)
    file_contents = readlines(output_file)
    BO_matrices = Matrix{Float64}[]
    for (i, line) in enumerate(file_contents)
        if occursin("Wiberg bond index matrix in the NAO basis:", line)
            line_idx = i + 4 # takes you to first line of matrix

            # find the number of atoms
            BO_line = file_contents[line_idx]
            natoms = 0
            while strip(BO_line) != ""
                line_idx += 1
                BO_line = file_contents[line_idx]
                natoms += 1
            end

            # number of blocks over which BO matrix is printed
            num_BO_blocks = natoms ÷ 9 + 1
            BO_matrix = zeros(natoms, natoms)

            # reset to first line of data
            line_idx = i + 4
            for i_block in 1:num_BO_blocks
                for i_atom in 1:natoms
                    BO_line = file_contents[line_idx]
                    split_line = split(BO_line)[3:end]
                    BOs = tryparse.((Float64,), split_line)
                    for j in eachindex(BOs)
                        # 9 is the number of atoms printed per block
                        BO_matrix[i_atom, (i_block-1)*9+j] = BOs[j]
                    end
                    line_idx += 1
                end
                line_idx += 3
            end
            @assert issymmetric(BO_matrix) "Bond order matrix is not symmetric. Must have parsed incorrectly."
            push!(BO_matrices, BO_matrix)
        end
    end
    return BO_matrices
end

"""
Parses EDA terms from an EDA calculation.
Returns a dictionary of arrays of each term.
"""
function parse_EDA_terms(output_file::String)
    cls_elec = Float64[]
    elec = Float64[]
    mod_pauli = Float64[]
    pauli = Float64[]
    disp = Float64[]
    pol = Float64[]
    ct = Float64[]

    lines = readlines(output_file)
    for line in lines
        if occursin("(ELEC)", line)
            push!(elec, tryparse(Float64, split(line)[5]))
        end
        if occursin("(PAULI)", line)
            push!(pauli, tryparse(Float64, split(line)[5]))
        end
        if occursin("E_disp   (DISP)", line)
            push!(disp, tryparse(Float64, split(line)[5]))
        end
        if occursin("(CLS ELEC)", line)
            push!(cls_elec, tryparse(Float64, split(line)[6]))
        end
        if occursin("(MOD PAULI)", line)
            push!(mod_pauli, tryparse(Float64, split(line)[6]))
        end
        if occursin("POLARIZATION", line)
            push!(pol, tryparse(Float64, split(line)[2]))
        end
        if occursin("CHARGE TRANSFER", line)
            push!(ct, tryparse(Float64, split(line)[3]))
        end
    end
    return Dict(
        :cls_elec => cls_elec,
        :elec => elec,
        :mod_pauli => mod_pauli,
        :pauli => pauli,
        :disp => disp,
        :pol => pol,
        :ct => ct
    )
end

"""
Parses EDA terms from an EDA calculation.
Appends output to existing dictionary.
"""
function parse_EDA_terms!(eda_dict::Dict{Symbol, Vector{Float64}}, output_file::String)
    lines = readlines(output_file)
    for line in lines
        if occursin("(ELEC)", line)
            push!(eda_dict[:elec], tryparse(Float64, split(line)[5]))
        end
        if occursin("(PAULI)", line)
            push!(eda_dict[:pauli], tryparse(Float64, split(line)[5]))
        end
        if occursin("E_disp   (DISP)", line)
            push!(eda_dict[:disp], tryparse(Float64, split(line)[5]))
        end
        if occursin("(CLS ELEC)", line)
            push!(eda_dict[:cls_elec], tryparse(Float64, split(line)[6]))
        end
        if occursin("(MOD PAULI)", line)
            push!(eda_dict[:mod_pauli], tryparse(Float64, split(line)[6]))
        end
        if occursin("POLARIZATION", line)
            push!(eda_dict[:pol], tryparse(Float64, split(line)[2]))
        end
        if occursin("CHARGE TRANSFER", line)
            push!(eda_dict[:ct], tryparse(Float64, split(line)[3]))
        end
    end
    return
end

"""
Parses chelpg charges from output file if they are there.
Returns as vector of vectors.
"""
function parse_chelpg_charges(output_file::String)
    charges = Vector{Float64}[]

    lines = readlines(output_file)
    for (i, line) in enumerate(lines)
        if occursin("Ground-State ChElPG Net Atomic Charges", line)
            new_charges = Float64[]
            line_index = i + 4
            while !occursin("------------", lines[line_index])
                push!(new_charges, tryparse(Float64, split(lines[line_index])[3]))
                line_index += 1
            end
            push!(charges, new_charges)
        end
    end
    return charges
end

"""
Parses the geometries used by Q-Chem in a calculation.
Generically, these may be different than what was input by
the user since Q-Chem will translate and rotate the molecule
to a standard orientation and center the molecule at its center of mass.
"""
function parse_geometries(output_file::String)
    labels = Vector{String}[]
    geometries = Matrix{Float64}[]

    lines = readlines(output_file)
    for (i, line) in enumerate(lines)
        if occursin("Standard Nuclear Orientation (Angstroms)", line)
            line_index = i + 3
            natoms = 0
            while !occursin("------------", lines[line_index])
                natoms += 1
                line_index += 1
            end
            line_index = i + 3
            new_labels = ["" for _ in 1:natoms]
            new_geom = zeros(3, natoms)
            for i_geom in 1:natoms
                split_line = split(lines[line_index])
                new_labels[i_geom] = split_line[2]
                @views new_geom[:, i_geom] = tryparse.((Float64,), split_line[3:5])
                line_index += 1
            end
            push!(labels, new_labels)
            push!(geometries, new_geom)
        end
    end
    return labels, geometries
end

"""
Parses the geometries used by Q-Chem in a calculation.
Generically, these may be different than what was input by
the user since Q-Chem will translate and rotate the molecule
to a standard orientation and center the molecule at its center of mass.
Also finds the corresponding calculated energy. Assumes DFT.
"""
function parse_geometries_and_energies(output_file::String)
    labels = Vector{String}[]
    geometries = Matrix{Float64}[]
    energies = Float64[]

    lines = readlines(output_file)
    for (i, line) in enumerate(lines)
        if occursin("Standard Nuclear Orientation (Angstroms)", line)
            line_index = i + 3
            natoms = 0
            while !occursin("------------", lines[line_index])
                natoms += 1
                line_index += 1
            end
            line_index = i + 3
            new_labels = ["" for _ in 1:natoms]
            new_geom = zeros(3, natoms)
            for i_geom in 1:natoms
                split_line = split(lines[line_index])
                new_labels[i_geom] = split_line[2]
                @views new_geom[:, i_geom] = tryparse.((Float64,), split_line[3:5])
                line_index += 1
            end
            push!(labels, new_labels)
            push!(geometries, new_geom)
            # Now find the corresponding energy
            while !occursin("Total energy", lines[line_index]) && (line_index <= length(lines))
                line_index += 1
            end
            energy = tryparse(Float64, split(lines[line_index])[end])
            push!(energies, energy)
        end
    end
    return energies, labels, geometries
end

"""
Parses xyz trajectory from Q-Chem aimd run.
Only writes gometry if the step is successfully completed.
"""
function get_xyz_trajectory_from_aimd_run(output_file::String)
    labels = Vector{String}[]
    geometries = Matrix{Float64}[]
    energies = Float64[]

    lines = readlines(output_file)
    added_geom = false
    for (i, line) in enumerate(lines)
        if occursin("Instantaneous Temperature", line)
            added_geom = true
            line_index = i + 5
            natoms = 0
            while !occursin("------------", lines[line_index])
                natoms += 1
                line_index += 1
            end
            line_index = i + 5
            new_labels = ["" for _ in 1:natoms]
            new_geom = zeros(3, natoms)
            for i_geom in 1:natoms
                split_line = split(lines[line_index])
                new_labels[i_geom] = split_line[2]
                @views new_geom[:, i_geom] = tryparse.((Float64,), split_line[3:5])
                line_index += 1
            end
            push!(labels, new_labels)
            push!(geometries, new_geom)
        end
        if occursin("Total energy in the final basis set", line) && added_geom
            push!(energies, tryparse(Float64, split(line)[9]))
            added_geom = false
        end
    end
    if added_geom
        # only get here if we added a geometry but never found a corresponding energy
        pop!(labels)
        pop!(geometries)
    end
    @assert length(energies) == length(geometries) "Geometries and energies don't pair up!"
    return energies, labels, geometries
end

"""
Takes all EDA outputs from a file and writes them in CSV format.
Additionally, we attach the file containing charges calculated with ChElPG
from which the charges and geometries can be parsed. We also store the index
of the geometry to which the EDA data corresponds.
"""
function write_eda_and_charge_data_to_csv(csv_out_file::String, eda_file::String, charge_and_geom_file::String)
    eda_data = parse_EDA_terms(eda_file)
    display(length(eda_data))
    df = DataFrame(eda_data)
    geom_index = [1:nrow(df)...]
    println(length([charge_and_geom_file for _ in eachrow(df)]))
    df[!, :index] = geom_index
    df[!, :charge_file] = [charge_and_geom_file for _ in eachrow(df)]
    CSV.write(csv_out_file, df)
end

function parse_xyz_from_EDA_input(infile::String)
    lines = readlines(infile)
    labels = String[]
    coords = Vector{Float64}[]
    in_molecule_block = false
    for line in lines
        if in_molecule_block
            split_line = split(line)
            if length(split_line) == 4
                if all(isletter, strip(split_line[1], ('+', '-')))
                    xyz = tryparse.((Float64,), split_line[2:4])
                    if !any(isnothing, xyz)
                        push!(coords, xyz)
                        push!(labels, strip(split_line[1], ('+', '-')))
                    end
                end
            end
        end
        if occursin("\$molecule", line)
            in_molecule_block = true
        end
        if occursin("\$end", line)
            return labels, reduce(hcat, coords)
        end
    end
    return nothing
end

"""
Reads all file names from within a path and finds the pairs of .in and .out files
which are expected to have EDA data. Parses the xyz file from the .in file and
writes th EDA data to a CSV file which contains EDA data for the entire scan.
"""
function write_xyz_and_csv_from_EDA_scans(folder_path::String, csv_outfile::String, xyz_outfile::String)
    all_files = readdir(folder_path)
    in_files = String[]
    for i in eachindex(all_files)
        if contains(all_files[i], ".in")
            push!(in_files, all_files[i])
        end
    end
    out_files = String[]
    for i in eachindex(in_files)
        out_file = replace(in_files[i], ".in" => ".out")
        if out_file in all_files
            push!(out_files, out_file)
        else
            @warn string("Couldn't find an out file corresponding to the input file: ", in_files[i])
            continue
        end
    end

    # get xyz coordinates from input files
    all_labels = Vector{String}[]
    all_geoms = Matrix{Float64}[]
    for i in eachindex(in_files)
        labels, geom = parse_xyz_from_EDA_input(in_files[i])
        push!(all_labels, labels)
        push!(all_geoms, geom)
    end

    # get EDA data from output files
    eda_data = Dict(
        :cls_elec => Float64[],
        :elec => Float64[],
        :mod_pauli => Float64[],
        :pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[]
    )

    for i in eachindex(out_files)
        parse_EDA_terms!(eda_data, out_files[i])
    end
    column_lengths = [length(eda_data[key]) for key in keys(eda_data)]
    if !allequal(column_lengths)
        for i in eachindex(column_lengths)
            if column_lengths[i] == 0
                key = [keys(eda_data)...][i]
                [push!(eda_data[key], 0.0) for _ in 1:maximum(column_lengths)]
            end
        end
    end
    column_lengths = [length(eda_data[key]) for key in keys(eda_data)]
    @assert allequal(column_lengths) "All columns of EDA data don't have equal length. Parsing failed."
    df = DataFrame(eda_data)
    geom_index = [1:nrow(df)...]
    df[!, :index] = geom_index
    write_xyz(xyz_outfile, [string(length(all_labels[i]), "\n") for i in eachindex(all_labels)], all_labels, all_geoms)
    df[!, :xyz_file] = [xyz_outfile for _ in eachrow(df)]
    CSV.write(csv_outfile, df)
    return
end