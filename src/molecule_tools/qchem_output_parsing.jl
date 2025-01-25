using LinearAlgebra, CSV, DataFrames, Combinatorics, ProgressBars, ProgressMeter
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
        if occursin("E_cls_disp  (CLS DISP)", line)
            push!(disp, tryparse(Float64, split(line)[6]))
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
function parse_EDA_terms!(eda_dict::Dict{Symbol, Vector{Float64}}, output_file::String, parse_fragment_energies::Bool=true, fragment_zero::Float64=0.0)
    lines = readlines(output_file)
    for (i, line) in enumerate(lines)
        if occursin("(ELEC)", line) && haskey(eda_dict, :elec)
            push!(eda_dict[:elec], tryparse(Float64, split(line)[5]))
        elseif occursin("(PAULI)", line) && haskey(eda_dict, :pauli)
            push!(eda_dict[:pauli], tryparse(Float64, split(line)[5]))
        elseif occursin("E_disp   (DISP)", line) && haskey(eda_dict, :disp)
            push!(eda_dict[:disp], tryparse(Float64, split(line)[5]))
        elseif occursin("E_cls_disp  (CLS DISP)", line) && haskey(eda_dict, :disp)
            push!(eda_dict[:disp], tryparse(Float64, split(line)[6]))
        elseif occursin("(CLS ELEC)", line) && haskey(eda_dict, :cls_elec)
            push!(eda_dict[:cls_elec], tryparse(Float64, split(line)[6]))
        elseif occursin("(MOD PAULI)", line) && haskey(eda_dict, :mod_pauli)
            push!(eda_dict[:mod_pauli], tryparse(Float64, split(line)[6]))
        elseif occursin("POLARIZATION", line) && haskey(eda_dict, :pol)
            push!(eda_dict[:pol], tryparse(Float64, split(line)[2]))
        elseif occursin("CHARGE TRANSFER", line) && haskey(eda_dict, :ct)
            push!(eda_dict[:ct], tryparse(Float64, split(line)[3]))
        elseif occursin("Fragment Energies", line)
            if parse_fragment_energies && haskey(eda_dict, :deform)
                index = copy(i+1)
                fragment_sum = 0.0
                num_fragments = 0
                while !occursin("--------", lines[index])
                    fragment_sum += tryparse(Float64, split(lines[index])[2])
                    index += 1
                    num_fragments += 1
                end
                if fragment_zero == 0.0
                    push!(eda_dict[:deform], fragment_sum)
                else
                    push!(eda_dict[:deform], (fragment_sum - num_fragments * fragment_zero) * 627.51 * 4.184)
                end
            end
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

function parse_final_geometry_from_output_files(output_files::Vector{String})
    all_labels = Vector{String}[]
    all_geoms = Matrix{Float64}[]
    for i in ProgressBar(eachindex(output_files))
        if occursin(".out", output_files[i])
            labels, geoms = parse_geometries(output_files[i])
            push!(all_labels, labels[end])
            push!(all_geoms, geoms[end])
        end
    end
    return all_labels, all_geoms
end

"""
Parses the geometry from a Q-Chem optimization.
"""
function parse_geometry_optimization(output_file::String)
    lines = readlines(output_file)
    for (i, line) in enumerate(lines)
        if occursin("CONVERGED", line)
            line_index = i + 6
            natoms = 0
            while !occursin("------------", lines[line_index])
                natoms += 1
                line_index += 1
            end
            line_index = i + 6
            labels = ["" for _ in 1:natoms]
            geom = zeros(3, natoms)
            for i_geom in 1:natoms
                split_line = split(lines[line_index])
                labels[i_geom] = split_line[2]
                @views geom[:, i_geom] = tryparse.((Float64,), split_line[3:5])
                line_index += 1
            end
            return labels, geom
        end
    end
end

function parse_relaxed_scan(output_file::String)
    lines = readlines(output_file)
    all_labels = Vector{String}[]
    all_geoms = Matrix{Float64}[]
    for (i, line) in enumerate(lines)
        if occursin("CONVERGED", line)
            line_index = i + 5
            natoms = 0
            while !occursin("Z-matrix", lines[line_index])
                natoms += 1
                line_index += 1
            end
            natoms -= 1 # we count a blank line before getting to Z-matrix
            line_index = i + 5
            labels = ["" for _ in 1:natoms]
            geom = zeros(3, natoms)
            for i_geom in 1:natoms
                split_line = split(lines[line_index])
                labels[i_geom] = split_line[2]
                @views geom[:, i_geom] = tryparse.((Float64,), split_line[3:5])
                line_index += 1
            end
            push!(all_labels, labels)
            push!(all_geoms, geom)
        end
    end
    return all_labels, all_geoms
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
            # Now find the corresponding energy
            while !occursin("Total energy", lines[line_index]) && (line_index <= length(lines))
                line_index += 1
            end
            energy = tryparse(Float64, split(lines[line_index])[end])
            push!(labels, new_labels)
            push!(geometries, new_geom)
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

function parse_xyz_and_eda_from_output!(infile::String, eda_dict::Dict{Symbol, Vector{Float64}}, parse_fragment_energies::Bool=true, fragment_zero::Float64=0.0)
    lines = readlines(infile)

    labels = String[]
    coords = Vector{Float64}[]
    pending_coords = Matrix{Float64}(undef, 0, 0)
    pending_labels = String[]
    final_labels = Vector{String}[]
    final_coords = Matrix{Float64}[]

    successfully_parsed_coords = false
    successfully_parsed_eda    = false

    in_molecule_block = false
    found_fragment_separator = false
    for (i, line) in enumerate(lines)
        if in_molecule_block
            if successfully_parsed_coords
                # we can only get here if we parsed some coordinates
                # but then failed to find eda terms corresponding to
                # that geometry
                successfully_parsed_coords = false
            end
            split_line = split(line)
            if length(split_line) == 4 && found_fragment_separator
                if all(isletter, strip(split_line[1], ('+', '-')))
                    xyz = tryparse.((Float64,), split_line[2:4])
                    if !any(isnothing, xyz)
                        push!(coords, xyz)
                        push!(labels, strip(split_line[1], ('+', '-')))
                    end
                end
            elseif found_fragment_separator == false
                if occursin("--", line)
                    found_fragment_separator = true
                end
            end
        end
        if occursin("\$molecule", line) && !successfully_parsed_coords
            in_molecule_block = true
        end
        if occursin("\$end", line) && in_molecule_block
            if found_fragment_separator
                # If we get here, we successfully parsed a geometry.
                # We then store these coordinates and labels in the 
                # pending and say we should keep this geometry if we
                # successfully parsed some coordinates and
                # later successfully parse eda values.
                pending_labels = copy(labels)
                pending_coords = reduce(hcat, coords)
                successfully_parsed_coords = true
            end
            in_molecule_block = false
            found_fragment_separator = false
            labels = String[]
            coords = Vector{Float64}[]
        end
        if occursin("(ELEC)", line) && haskey(eda_dict, :elec)
            push!(eda_dict[:elec], tryparse(Float64, split(line)[5]))
        elseif occursin("(PAULI)", line) && haskey(eda_dict, :pauli)
            push!(eda_dict[:pauli], tryparse(Float64, split(line)[5]))
        elseif occursin("E_disp   (DISP)", line) && haskey(eda_dict, :disp)
            push!(eda_dict[:disp], tryparse(Float64, split(line)[5]))
        elseif occursin("E_cls_disp  (CLS DISP)", line) && haskey(eda_dict, :disp)
            push!(eda_dict[:disp], tryparse(Float64, split(line)[6]))
        elseif occursin("(CLS ELEC)", line) && haskey(eda_dict, :cls_elec)
            push!(eda_dict[:cls_elec], tryparse(Float64, split(line)[6]))
        elseif occursin("(MOD PAULI)", line) && haskey(eda_dict, :mod_pauli)
            push!(eda_dict[:mod_pauli], tryparse(Float64, split(line)[6]))
        elseif occursin("POLARIZATION", line) && haskey(eda_dict, :pol)
            push!(eda_dict[:pol], tryparse(Float64, split(line)[2]))
        elseif occursin("CHARGE TRANSFER", line) && haskey(eda_dict, :ct)
            push!(eda_dict[:ct], tryparse(Float64, split(line)[3]))
            # store the parsed coordinates because we found all the
            # corresponding EDA terms
            if successfully_parsed_coords
                push!(final_labels, pending_labels)
                push!(final_coords, pending_coords)
                successfully_parsed_coords = false
                successfully_parsed_eda    = false
                num_labels = length(final_labels)
                num_eda_terms = length(eda_dict[:ct])
                @assert length(final_labels) == length(eda_dict[:ct]) "Number of parsed geometries ($num_labels) and nummber of parsed eda terms ($num_eda_terms) aren't equal! Something went wrong with parsing."
            end
        elseif occursin("Fragment Energies", line)
            if parse_fragment_energies && haskey(eda_dict, :deform)
                index = copy(i+1)
                fragment_sum = 0.0
                num_fragments = 0
                while !occursin("--------", lines[index])
                    fragment_sum += tryparse(Float64, split(lines[index])[2])
                    index += 1
                    num_fragments += 1
                end
                push!(eda_dict[:deform], (fragment_sum - num_fragments * fragment_zero) * 627.51 * 4.184)
            end
        end
        if occursin("fatal error", line) && successfully_parsed_coords
            @warn string("Found failed job corresponding to job input ", length(final_labels), ". Throwing away the geometry and continuing.")
            successfully_parsed_coords = false
        end
    end
    num_labels = length(final_labels)
    num_eda_terms = length(eda_dict[:ct])
    @assert length(final_labels) == length(eda_dict[:ct]) "Number of parsed geometries ($num_labels) and nummber of parsed eda terms ($num_eda_terms) aren't equal! Something went wrong with parsing."
    return final_labels, final_coords
end

function parse_xyz_and_fda_data(infile::String)
    lines = readlines(infile)

    labels = String[]
    coords = Vector{Float64}[]
    final_coords = Matrix{Float64}(undef, 0, 0)
    final_labels = String[]

    natoms = 0

    temp_forces = Vector{Float64}[]
    final_deformation_forces = Matrix{Float64}(undef, 0, 0)
    final_non_elec_frozen_forces = Matrix{Float64}(undef, 0, 0)
    final_polarization_forces = Matrix{Float64}(undef, 0, 0)
    final_charge_transfer_forces = Matrix{Float64}(undef, 0, 0)
    final_electrostatic_forces = Matrix{Float64}(undef, 0, 0)
    final_total_forces = Matrix{Float64}(undef, 0, 0)

    successfully_parsed_coords = false

    in_molecule_block = false
    found_fragment_separator = false
    for (i, line) in enumerate(lines)
        if in_molecule_block
            if successfully_parsed_coords
                # we can only get here if we parsed some coordinates
                # but then failed to find eda terms corresponding to
                # that geometry
                successfully_parsed_coords = false
            end
            split_line = split(line)
            if length(split_line) == 4 && found_fragment_separator
                if all(isletter, strip(split_line[1], ('+', '-')))
                    xyz = tryparse.((Float64,), split_line[2:4])
                    if !any(isnothing, xyz)
                        push!(coords, xyz)
                        push!(labels, strip(split_line[1], ('+', '-')))
                    end
                end
            elseif found_fragment_separator == false
                if occursin("--", line)
                    found_fragment_separator = true
                end
            end
        end
        if occursin("\$molecule", line) && !successfully_parsed_coords
            in_molecule_block = true
        end
        if occursin("\$end", line) && in_molecule_block
            if found_fragment_separator
                # If we get here, we successfully parsed a geometry.
                # We then store these coordinates and labels in the 
                # pending and say we should keep this geometry if we
                # successfully parsed some coordinates and
                # later successfully parse eda values.
                final_labels = copy(labels)
                final_coords = reduce(hcat, coords)
                natoms = length(final_labels)
                successfully_parsed_coords = true
            end
            in_molecule_block = false
            found_fragment_separator = false
        end
        if occursin("Geometric Distortion Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_deformation_forces = reduce(hcat, temp_forces)
            empty!(temp_forces)
        elseif occursin("Classical Electrostatic Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_electrostatic_forces = reduce(hcat, temp_forces)
            empty!(temp_forces)
        elseif occursin("Non-Electrostatic Frozen Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_non_elec_frozen_forces = reduce(hcat, temp_forces)
            empty!(temp_forces)
        elseif occursin("Polarization Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_polarization_forces = reduce(hcat, temp_forces)
            empty!(temp_forces)
        elseif occursin("Charge Transfer Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_charge_transfer_forces = reduce(hcat, temp_forces)
            empty!(temp_forces)
        elseif occursin("Total Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_total_forces = reduce(hcat, temp_forces)
            empty!(temp_forces)
        end
        if occursin("fatal error", line) && successfully_parsed_coords
            @warn string("Found failed job corresponding to job input ", length(final_labels), ". Throwing away the geometry and continuing.")
            successfully_parsed_coords = false
        end
    end
    force_dict = Dict{Symbol, Matrix{Float64}}(
        :Deformation => final_deformation_forces,
        :NonElecFrozen => final_non_elec_frozen_forces,
        :Electrostatics => final_electrostatic_forces,
        :Polarization => final_polarization_forces,
        :ChargeTransfer => final_charge_transfer_forces,
        :Total => final_total_forces
    )
    return final_labels, final_coords, force_dict
end

function parse_multiple_xyz_and_fda_data(infile::String)
    lines = readlines(infile)

    labels = String[]
    coords = Vector{Float64}[]
    final_coords = Matrix{Float64}[]
    final_labels = Vector{String}[]

    natoms = 0

    temp_forces = Vector{Float64}[]
    final_deformation_forces = Matrix{Float64}(undef, 0, 0)
    final_non_elec_frozen_forces = Matrix{Float64}(undef, 0, 0)
    final_polarization_forces = Matrix{Float64}(undef, 0, 0)
    final_charge_transfer_forces = Matrix{Float64}(undef, 0, 0)
    final_electrostatic_forces = Matrix{Float64}(undef, 0, 0)
    final_total_forces = Matrix{Float64}(undef, 0, 0)

    force_dict = Dict{Symbol, Vector{Matrix{Float64}}}(
        :Deformation => Matrix{Float64}[],
        :NonElecFrozen => Matrix{Float64}[],
        :Electrostatics => Matrix{Float64}[],
        :Polarization => Matrix{Float64}[],
        :ChargeTransfer => Matrix{Float64}[],
        :Total => Matrix{Float64}[]
    )

    successfully_parsed_coords = false

    in_molecule_block = false
    found_fragment_separator = false
    for (i, line) in enumerate(lines)
        if in_molecule_block
            if successfully_parsed_coords
                # we can only get here if we parsed some coordinates
                # but then failed to find eda terms corresponding to
                # that geometry
                successfully_parsed_coords = false
            end
            split_line = split(line)
            if length(split_line) == 4 && found_fragment_separator
                if all(isletter, strip(split_line[1], ('+', '-')))
                    xyz = tryparse.((Float64,), split_line[2:4])
                    if !any(isnothing, xyz)
                        push!(coords, xyz)
                        push!(labels, strip(split_line[1], ('+', '-')))
                    end
                end
            elseif found_fragment_separator == false
                if occursin("--", line)
                    found_fragment_separator = true
                end
            end
        end
        if occursin("\$molecule", line) && !successfully_parsed_coords
            in_molecule_block = true
        end
        if occursin("\$end", line) && in_molecule_block
            if found_fragment_separator
                # If we get here, we successfully parsed a geometry.
                # We then store these coordinates and labels in the 
                # pending and say we should keep this geometry if we
                # successfully parsed some coordinates and
                # later successfully parse eda values.
                push!(final_labels, copy(labels))
                push!(final_coords, reduce(hcat, coords))
                natoms = length(final_labels[end])
                successfully_parsed_coords = true
                empty!(labels)
                empty!(coords)
            end
            in_molecule_block = false
            found_fragment_separator = false
        end
        if occursin("Geometric Distortion Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_deformation_forces = reduce(hcat, temp_forces)
            push!(force_dict[:Deformation], final_deformation_forces)
            empty!(temp_forces)
        elseif occursin("Classical Electrostatic Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_electrostatic_forces = reduce(hcat, temp_forces)
            push!(force_dict[:Electrostatics], final_electrostatic_forces)
            empty!(temp_forces)
        elseif occursin("Non-Electrostatic Frozen Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_non_elec_frozen_forces = reduce(hcat, temp_forces)
            push!(force_dict[:NonElecFrozen], final_non_elec_frozen_forces)
            empty!(temp_forces)
        elseif occursin("Polarization Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_polarization_forces = reduce(hcat, temp_forces)
            push!(force_dict[:Polarization], final_polarization_forces)
            empty!(temp_forces)
        elseif occursin("Charge Transfer Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            final_charge_transfer_forces = reduce(hcat, temp_forces)
            push!(force_dict[:ChargeTransfer], final_charge_transfer_forces)
            empty!(temp_forces)
        elseif occursin("Total Forces", line)
            for j in 1:natoms
                split_line = split(lines[i+j+1])
                xyz = tryparse.((Float64,), split_line[2:4])
                if !any(isnothing, xyz)
                    push!(temp_forces, xyz)
                end
            end
            # reset parsing state since this is final set of forces for this input #
            successfully_parsed_coords = false
            
            final_total_forces = reduce(hcat, temp_forces)
            push!(force_dict[:Total], final_total_forces)
            empty!(temp_forces)
        end
        if occursin("fatal error", line) && successfully_parsed_coords
            @warn string("Found failed job corresponding to job input ", length(final_labels), ". Throwing away the geometry and continuing.")
            successfully_parsed_coords = false
            pop!(final_labels)
            pop!(final_coords)
        end
    end
    return final_labels, final_coords, force_dict
end

function parse_polarizabilities_and_energies_from_outfile(file::String)
    lines = readlines(file)

    all_energies = Float64[]
    actual_energies = Float64[]
    polarizabilities = Matrix{Float64}[]
    for i in eachindex(lines)
        if occursin("Polarizability tensor      [a.u.]", lines[i])
            α = zeros(3, 3)
            row_1 = parse.((Float64,), split(lines[i+1]))
            row_2 = parse.((Float64,), split(lines[i+2]))
            row_3 = parse.((Float64,), split(lines[i+3]))
            α[1, 1] = row_1[1]
            α[1, 2] = row_1[2]
            α[1, 3] = row_1[3]
            α[2, 1] = row_2[1]
            α[2, 2] = row_2[2]
            α[2, 3] = row_2[3]
            α[3, 1] = row_3[1]
            α[3, 2] = row_3[2]
            α[3, 3] = row_3[3]
            push!(polarizabilities, α)
        end
        if occursin("Total energy", lines[i])
            push!(all_energies, parse(Float64, split(lines[i])[9]))
        end
    end
    @assert length(all_energies) == 6 * length(polarizabilities) "Number of energies is not 6 times number of polarizabilities. Probably need to update code to work with analytic polarizability calculations."

    for i in 1:6:length(all_energies)
        E_estimate_1 = 0.5 * (all_energies[i] + all_energies[i+1])
        E_estimate_2 = 0.5 * (all_energies[i+2] + all_energies[i+3])
        E_estimate_3 = 0.5 * (all_energies[i+4] + all_energies[i+5])
        push!(actual_energies, (E_estimate_1 + E_estimate_2 + E_estimate_3) / 3.0)
    end

    @assert length(actual_energies) == length(polarizabilities)

    return actual_energies, polarizabilities
end

function parse_polarizabilities_energies_and_distances_from_outfile(file::String)
    lines = readlines(file)

    actual_energies = Float64[]
    polarizabilities = Matrix{Float64}[]
    distances = Float64[]

    temp_distances = Float64[]
    temp_energies = Float64[]
    for i in eachindex(lines)
        if occursin("Polarizability tensor      [a.u.]", lines[i])
            α = zeros(3, 3)
            row_1 = parse.((Float64,), split(lines[i+1]))
            row_2 = parse.((Float64,), split(lines[i+2]))
            row_3 = parse.((Float64,), split(lines[i+3]))
            α[1, 1] = row_1[1]
            α[1, 2] = row_1[2]
            α[1, 3] = row_1[3]
            α[2, 1] = row_2[1]
            α[2, 2] = row_2[2]
            α[2, 3] = row_2[3]
            α[3, 1] = row_3[1]
            α[3, 2] = row_3[2]
            α[3, 3] = row_3[3]

            # Check that no intermediate calculations failed. If they did,
            # we just don't store any of the parsed data and move on to
            # the next set of polarizability caluculations.
            if length(temp_distances) == 6 && length(temp_energies) == 6
                push!(polarizabilities, α)
                for i in 1:6:length(temp_energies)
                    E_estimate_1 = 0.5 * (temp_energies[1] + temp_energies[2])
                    E_estimate_2 = 0.5 * (temp_energies[3] + temp_energies[4])
                    E_estimate_3 = 0.5 * (temp_energies[5] + temp_energies[6])
                    push!(actual_energies, (E_estimate_1 + E_estimate_2 + E_estimate_3) / 3.0)
                end
                push!(distances, temp_distances[1])
            end
            temp_distances = Float64[]
            temp_energies = Float64[]
        end
        if occursin("Total energy", lines[i])
            push!(temp_energies, parse(Float64, split(lines[i])[9]))
        end
        if occursin("Distance Matrix", lines[i])
            push!(temp_distances, parse(Float64, split(lines[i+2])[end]))
        end
    end

    @assert length(actual_energies) == length(polarizabilities)
    @assert length(distances) == length(polarizabilities)

    return distances, actual_energies, polarizabilities
end

function write_coords_and_forces(file_prefix::String, coords::Vector{Matrix{Float64}}, labels::Vector{Vector{String}}, fda_dict::Dict{Symbol, Vector{Matrix{Float64}}})
    write_xyz(string(file_prefix, ".xyz"), labels, coords)
    write_xyz(string(file_prefix, "_deformation_forces.xyz"), labels, fda_dict[:Deformation])
    write_xyz(string(file_prefix, "_pol_forces.xyz"), labels, fda_dict[:Polarization])
    write_xyz(string(file_prefix, "_elec_forces.xyz"), labels, fda_dict[:Electrostatics])
    write_xyz(string(file_prefix, "_non_elec_frozen_forces.xyz"), labels, fda_dict[:NonElecFrozen])
    write_xyz(string(file_prefix, "_ct_forces.xyz"), labels, fda_dict[:ChargeTransfer])
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

function write_xyz_and_csv_from_EDA_calculation(eda_job_output_file::String, csv_outfile::String, xyz_outfile::String, dist_index_1::Int=0, dist_index_2::Int=0, zero_ct_beyond_distance::Float64=0.0)

    # NOTE: For molecules that use ECPs, the regular Pauli and Elec are not calculated.
    # For simplicity, we ignore them here. But if you uncomment them and are parsing
    # for molecules that don't use ECPs, they will be parsed properly.
    eda_data = Dict(
        #:elec => Float64[],
        #:pauli => Float64[],
        :cls_elec => Float64[],
        :mod_pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[],
        :int => Float64[],
    )

    # get EDA data and xyz coordinates from output file
    labels, geoms = parse_xyz_and_eda_from_output!(eda_job_output_file, eda_data)
    # populate interaction energy key
    for i in eachindex(eda_data[:cls_elec])
        push!(eda_data[:int], eda_data[:cls_elec][i] + eda_data[:mod_pauli][i] + eda_data[:ct][i] + eda_data[:disp][i] + eda_data[:pol][i])
    end
    column_lengths = [length(eda_data[key]) for key in keys(eda_data)]
    @assert allequal(column_lengths) "All columns of EDA data don't have equal length. Parsing failed."

    df = DataFrame(eda_data)
    geom_index = [1:nrow(df)...]
    df[!, :index] = geom_index

    if dist_index_1 != 0 && dist_index_2 != 0
        distances = zeros(length(geoms))
        for i in eachindex(geoms)
            r = norm(geoms[i][:, dist_index_1] - geoms[i][:, dist_index_2])
            distances[i] = r
            if zero_ct_beyond_distance > 0.0
                if r >= zero_ct_beyond_distance
                    df[!, :int][i] -= df[!, :ct][i]
                    df[!, :ct][i] = 0.0
                end
            end
        end
        df[!, :distances] = distances
    end

    write_xyz(xyz_outfile, [string(length(labels[i]), "\n") for i in eachindex(labels)], labels, geoms)
    CSV.write(csv_outfile, df)
    return
end

"""
Takes a full EDA calculation and the calculations on corresponding
many-body subsystems. Returns a dictionary of each term at each
many-body level and the full system. Also returns the full system geometry.
"""
function process_EDA_mbe_calculation(full_output_file::String, mbe_output_files::String...)

    E_h2o = -76.440791829812
    # get EDA data from output files
    total_eda_data = Dict(
        :cls_elec => Float64[],
        :elec => Float64[],
        :mod_pauli => Float64[],
        :pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[],
        :int => Float64[],
        :deform => Float64[],
        #:total => Float64[]
    )

    for mbe_file in mbe_output_files
        mbe_eda_data = Dict(
            :cls_elec => Float64[],
            :elec => Float64[],
            :mod_pauli => Float64[],
            :pauli => Float64[],
            :disp => Float64[],
            :pol => Float64[],
            :ct => Float64[],
            :int => Float64[],
            :deform => Float64[]
        )
        parse_EDA_terms!(mbe_eda_data, mbe_file, false, 0.0)
        for key in keys(mbe_eda_data)
            if key != :int
                term_total = sum(mbe_eda_data[key])
                push!(total_eda_data[key], term_total / 4.184)
            end
        end
        total_interaction = (
            total_eda_data[:cls_elec][end] +
            total_eda_data[:mod_pauli][end] +
            total_eda_data[:disp][end] +
            total_eda_data[:pol][end] +
            total_eda_data[:ct][end]
        )
        push!(total_eda_data[:int], total_interaction)
    end
    parse_EDA_terms!(total_eda_data, full_output_file, false, 0.0)
    for key in keys(total_eda_data)
        if length(total_eda_data[key]) > 0 && key != :int
            total_eda_data[key][end] /= 4.184
        end
    end
    total_interaction = (
        total_eda_data[:cls_elec][end] +
        total_eda_data[:mod_pauli][end] +
        total_eda_data[:disp][end] +
        total_eda_data[:pol][end] +
        total_eda_data[:ct][end]
    )   
    push!(total_eda_data[:int], total_interaction)
    #for i in eachindex(total_eda_data[:int])
    #    if i < length(total_eda_data[:int])
    #        total_eda_data[:deform][i] = total_eda_data[:deform][end]
    #    end
    #    push!(total_eda_data[:total], total_eda_data[:int][i] + total_eda_data[:deform][end])
    #    # ^^ Always add the end of deformation since the MBE one gets summed too many times
    #    # and the deformation at an MBE level and full level are always the same.
    #end

    # append total many-body contribution to each term
    for key in keys(total_eda_data)
        push!(total_eda_data[key], total_eda_data[key][end] - total_eda_data[key][1])
    end

    eda_data = Dict(
        :cls_elec => Float64[],
        :elec => Float64[],
        :mod_pauli => Float64[],
        :pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[]
    )

    labels, coords = parse_xyz_and_eda_from_output!(full_output_file, eda_data)
    return labels[1], [MVector{3, Float64}(coords[1][:, i]) for i in eachindex(eachcol(coords[1]))], total_eda_data
end

"""
file_prefix is whatever name comes before an integer. The suffix is assumed to be
full_system for the full systems and 2_body for the dimer calculations.
"""
function parse_two_body_many_body_eda_scan_and_write_geoms_and_data(output_file_prefix::String, num_scan_steps::Int)

    all_labels = Vector{String}[]
    all_geoms = Matrix{Float64}[]
    all_eda_data = Dict{Symbol, Vector{Float64}}[]
    for i in 1:num_scan_steps
        full_system_file = string(output_file_prefix, "_", i, "_full_system.out")
        two_body_file = string(output_file_prefix, "_", i, "_2_body.out")
        labels, geoms, eda_data = process_EDA_mbe_calculation(full_system_file, two_body_file)
        push!(all_labels, labels)
        push!(all_geoms, reduce(hcat, geoms))
        push!(all_eda_data, eda_data)
    end

    eda_data_as_df = convert_parsed_eda_data_to_linear_format(all_eda_data)
    write_xyz(string(output_file_prefix, "_geoms.xyz"), all_labels, all_geoms)
    CSV.write(string(output_file_prefix, ".csv"), eda_data_as_df)
end

function convert_parsed_eda_data_to_linear_format(eda_data::Vector{Dict{Symbol, Vector{Float64}}})
    all_pauli_energies = zeros(length(eda_data))
    all_pauli_2_body_energies = zeros(length(eda_data))
    all_pauli_3_body_energies = zeros(length(eda_data))
    all_elec_energies = zeros(length(eda_data))
    all_disp_energies = zeros(length(eda_data))
    all_disp_2_body_energies = zeros(length(eda_data))
    all_disp_3_body_energies = zeros(length(eda_data))
    all_pol_energies = zeros(length(eda_data))
    all_pol_2_body_energies = zeros(length(eda_data))
    all_pol_3_body_energies = zeros(length(eda_data))
    all_ct_energies = zeros(length(eda_data))
    all_ct_2_body_energies = zeros(length(eda_data))
    all_ct_3_body_energies = zeros(length(eda_data))
    all_int_energies = zeros(length(eda_data))
    all_int_2_body_energies = zeros(length(eda_data))
    all_int_3_body_energies = zeros(length(eda_data))
    all_total_energies = zeros(length(eda_data))
    all_total_2_body_energies = zeros(length(eda_data))
    all_total_3_body_energies = zeros(length(eda_data))

    for i in eachindex(eda_data)
        all_pauli_energies[i] = eda_data[i][:mod_pauli][2]
        all_pauli_2_body_energies[i] = eda_data[i][:mod_pauli][1]
        all_pauli_3_body_energies[i] = eda_data[i][:mod_pauli][3]
        all_elec_energies[i] = eda_data[i][:cls_elec][1]
        all_disp_energies[i] = eda_data[i][:disp][2]
        all_disp_2_body_energies[i] = eda_data[i][:disp][1]
        all_disp_3_body_energies[i] = eda_data[i][:disp][3]
        all_pol_energies[i] = eda_data[i][:pol][2]
        all_pol_2_body_energies[i] = eda_data[i][:pol][1]
        all_pol_3_body_energies[i] = eda_data[i][:pol][3]
        all_ct_energies[i] = eda_data[i][:ct][2]
        all_ct_2_body_energies[i] = eda_data[i][:ct][1]
        all_ct_3_body_energies[i] = eda_data[i][:ct][3]
        all_int_energies[i] = eda_data[i][:int][2]
        all_int_2_body_energies[i] = eda_data[i][:int][1]
        all_int_3_body_energies[i] = eda_data[i][:int][3]
        #all_total_energies[i] = eda_data[i][:total][2]
        #all_total_2_body_energies[i] = eda_data[i][:total][1]
        #all_total_3_body_energies[i] = eda_data[i][:total][3]
    end

    df = DataFrame(
        :mod_pauli => all_pauli_energies,
        :mod_pauli_2b => all_pauli_2_body_energies,
        :mod_pauli_3b => all_pauli_3_body_energies,
        :cls_elec => all_elec_energies,
        :disp => all_disp_energies,
        :disp_2b => all_disp_2_body_energies,
        :disp_3b => all_disp_3_body_energies,
        :pol => all_pol_energies,
        :pol_2b => all_pol_2_body_energies,
        :pol_3b => all_pol_3_body_energies,
        :ct => all_ct_energies,
        :ct_2b => all_ct_2_body_energies,
        :ct_3b => all_ct_3_body_energies,
        :int => all_int_energies,
        :int_2b => all_int_2_body_energies,
        :int_3b => all_int_3_body_energies,
        #:total => all_total_energies,
        #:total_2b => all_total_2_body_energies,
        #:total_3b => all_total_3_body_energies,
    )

    return df

end

function process_EDA_mbe_ion_water_calculation(full_output_file::String, two_body_output_file::String)

    E_h2o = -76.440791829812
    # TODO: Get the rest of the ion energies at wB97X-V/def2-qzvppd ^^^

    eda_data = Dict(
        :cls_elec => Float64[],
        :elec => Float64[],
        :mod_pauli => Float64[],
        :pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[]
    )

    labels, coords = parse_xyz_and_eda_from_output!(full_output_file, eda_data)
    # Everything in here assumes there is a single ion and the rest is water and
    # that ion comes at the beginning of the xyz coordinates.
    num_fragments = ((length(labels[1]) - 1) ÷ 3) + 1

    num_ion_water_dimers    = binomial(num_fragments-1, 1)

    two_body_eda_data = Dict(
        :cls_elec => Float64[],
        :mod_pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[],
        :int => Float64[],
        :deform => Float64[],
        :total => Float64[]
    )

    # Parse two and three body contributions to EDA MBE #
    parse_EDA_terms!(two_body_eda_data, two_body_output_file, false)

    # Make tuples of ion-water and water-water #
    start_ww  = num_ion_water_dimers + 1
    
    elec_2body  = (sum(two_body_eda_data[:cls_elec][1:start_ww-1]), sum(two_body_eda_data[:cls_elec][start_ww:end]))
    pauli_2body = (sum(two_body_eda_data[:mod_pauli][1:start_ww-1]), sum(two_body_eda_data[:mod_pauli][start_ww:end]))
    disp_2body  = (sum(two_body_eda_data[:disp][1:start_ww-1]), sum(two_body_eda_data[:disp][start_ww:end]))
    pol_2body   = (sum(two_body_eda_data[:pol][1:start_ww-1]), sum(two_body_eda_data[:pol][start_ww:end]))
    ct_2body    = (sum(two_body_eda_data[:ct][1:start_ww-1]), sum(two_body_eda_data[:ct][start_ww:end]))
    
    elec_mb  = eda_data[:cls_elec][1] - sum(elec_2body)
    pauli_mb  = eda_data[:mod_pauli][1] - sum(pauli_2body)
    disp_mb  = eda_data[:disp][1] - sum(disp_2body)
    pol_mb  = eda_data[:pol][1] - sum(pol_2body)
    ct_mb  = eda_data[:ct][1] - sum(ct_2body)

    total_eda_data = Dict(
        :cls_elec => [elec_2body[1], elec_2body[2], elec_mb, eda_data[:cls_elec][1]] / 4.184,
        :mod_pauli => [pauli_2body[1], pauli_2body[2], pauli_mb, eda_data[:mod_pauli][1]] / 4.184,
        :disp => [disp_2body[1], disp_2body[2], disp_mb, eda_data[:disp][1]] / 4.184,
        :pol => [pol_2body[1], pol_2body[2], pol_mb, eda_data[:pol][1]] / 4.184,
        :ct => [ct_2body[1], ct_2body[2], ct_mb, eda_data[:ct][1]] / 4.184,
        :int => Float64[
            elec_2body[1] + pauli_2body[1] + disp_2body[1] + pol_2body[1] + ct_2body[1],
            elec_2body[2] + pauli_2body[2] + disp_2body[2] + pol_2body[2] + ct_2body[2],
            elec_mb[1] + pauli_mb[1] + disp_mb[1] + pol_mb[1] + ct_mb[1],
            eda_data[:cls_elec][1] + eda_data[:mod_pauli][1] + eda_data[:disp][1] + eda_data[:pol][1] + eda_data[:ct][1]
        ] / 4.184,
        :deform => Float64[],
        :total => Float64[]
    )

    # TODO: Can determine the totals by including the deformation energies. For now, I just leave
    # this out cause I mainly care about analyzing the many-body contributions to the various terms.
    #push!(total_eda_data[:total], )

    return labels[1], [MVector{3, Float64}(coords[1][:, i]) for i in eachindex(eachcol(coords[1]))], total_eda_data
end

function process_EDA_mbe_ion_water_calculation(full_output_file::String, two_body_output_file::String, three_body_output_file::String)

    E_h2o = -76.440791829812
    # TODO: Get the rest of the ion energies at wB97X-V/def2-qzvppd ^^^
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

    labels, coords = parse_xyz_and_eda_from_output!(full_output_file, eda_data)
    # Everything in here assumes there is a single ion and the rest is water and
    # that ion comes at the beginning of the xyz coordinates.
    num_fragments = ((length(labels[1]) - 1) ÷ 3) + 1

    num_ion_water_dimers    = binomial(num_fragments-1, 1)
    num_ion_water_trimers   = binomial(num_fragments-1, 2)

    two_body_eda_data = Dict(
        :cls_elec => Float64[],
        :mod_pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[],
        :int => Float64[],
        :deform => Float64[],
        :total => Float64[]
    )

    three_body_eda_data = Dict(
        :cls_elec => Float64[],
        :mod_pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[],
        :int => Float64[],
        :deform => Float64[],
        :total => Float64[]
    )

    # Parse two and three body contributions to EDA MBE #
    parse_EDA_terms!(two_body_eda_data, two_body_output_file, false)
    parse_EDA_terms!(three_body_eda_data, three_body_output_file, false)

    # Make tuples of ion-water and water-water #
    start_ww  = num_ion_water_dimers + 1
    
    elec_2body  = (sum(two_body_eda_data[:cls_elec][1:start_ww-1]), sum(two_body_eda_data[:cls_elec][start_ww:end]))
    pauli_2body = (sum(two_body_eda_data[:mod_pauli][1:start_ww-1]), sum(two_body_eda_data[:mod_pauli][start_ww:end]))
    disp_2body  = (sum(two_body_eda_data[:disp][1:start_ww-1]), sum(two_body_eda_data[:disp][start_ww:end]))
    pol_2body   = (sum(two_body_eda_data[:pol][1:start_ww-1]), sum(two_body_eda_data[:pol][start_ww:end]))
    ct_2body    = (sum(two_body_eda_data[:ct][1:start_ww-1]), sum(two_body_eda_data[:ct][start_ww:end]))
    
    # Make tuples of ion-water-water and water-water-water terms #
    # NOTE: It is easy to work out that the number of times each ion-water dimer
    # appears in the set of i-w-w triples is the number of waters minus one (for one ion).
    # Similarly, the number of times each water dimer appears is always one.
    # So, we take the sum of all ion-water trimer energies for each component and subtract
    # off the total 2-body water-water contribution and num_waters-1 times the ion-water
    # contribution. This leaves us with just the 3-body i-w-w contribution.
    # The w-w-w contributions are easily shown to have every dimer contibue num_waters-2 times.
    start_www = num_ion_water_trimers + 1
    num_waters = num_fragments - 1
    elec_3body  = (sum(three_body_eda_data[:cls_elec][1:start_www-1]) - (num_waters-1) * elec_2body[1] - elec_2body[2], sum(three_body_eda_data[:cls_elec][start_www:end]) - (num_waters-2) * elec_2body[2])
    pauli_3body  = (sum(three_body_eda_data[:mod_pauli][1:start_www-1]) - (num_waters-1) * pauli_2body[1] - pauli_2body[2], sum(three_body_eda_data[:mod_pauli][start_www:end]) - (num_waters-2) * pauli_2body[2])
    disp_3body  = (sum(three_body_eda_data[:disp][1:start_www-1]) - (num_waters-1) * disp_2body[1] - disp_2body[2], sum(three_body_eda_data[:disp][start_www:end]) - (num_waters-2) * disp_2body[2])
    pol_3body  = (sum(three_body_eda_data[:pol][1:start_www-1]) - (num_waters-1) * pol_2body[1] - pol_2body[2], sum(three_body_eda_data[:pol][start_www:end]) - (num_waters-2) * pol_2body[2])
    ct_3body  = (sum(three_body_eda_data[:ct][1:start_www-1]) - (num_waters-1) * ct_2body[1] - ct_2body[2], sum(three_body_eda_data[:ct][start_www:end]) - (num_waters-2) * ct_2body[2])

    total_eda_data = Dict(
        :cls_elec => [elec_2body[1], elec_2body[2], elec_3body[1], elec_3body[2], eda_data[:cls_elec][1]] / 4.184,
        :mod_pauli => [pauli_2body[1], pauli_2body[2], pauli_3body[1], pauli_3body[2], eda_data[:mod_pauli][1]] / 4.184,
        :disp => [disp_2body[1], disp_2body[2], disp_3body[1], disp_3body[2], eda_data[:disp][1]] / 4.184,
        :pol => [pol_2body[1], pol_2body[2], pol_3body[1], pol_3body[2], eda_data[:pol][1]] / 4.184,
        :ct => [ct_2body[1], ct_2body[2], ct_3body[1], ct_3body[2], eda_data[:ct][1]] / 4.184,
        :int => Float64[
            elec_2body[1] + pauli_2body[1] + disp_2body[1] + pol_2body[1] + ct_2body[1],
            elec_2body[2] + pauli_2body[2] + disp_2body[2] + pol_2body[2] + ct_2body[2],
            elec_3body[1] + pauli_3body[1] + disp_3body[1] + pol_3body[1] + ct_3body[1],
            elec_3body[2] + pauli_3body[2] + disp_3body[2] + pol_3body[2] + ct_3body[2],
            eda_data[:cls_elec][1] + eda_data[:mod_pauli][1] + eda_data[:disp][1] + eda_data[:pol][1] + eda_data[:ct][1]
        ] / 4.184,
        :deform => Float64[],
        :total => Float64[]
    )

    # TODO: Can determine the totals by including the deformation energies. For now, I just leave
    # this out cause I mainly care about analyzing the many-body contributions to the various terms.
    #push!(total_eda_data[:total], )

    return labels[1], [MVector{3, Float64}(coords[1][:, i]) for i in eachindex(eachcol(coords[1]))], total_eda_data
end

function process_EDA_mbe_ion_water_calculation_into_all_many_body_terms(full_output_file::String, two_body_output_file::String, three_body_output_file::String)

    eda_data = Dict(
        :cls_elec => Float64[],
        :mod_pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[]
    )
    dimer_labels, dimer_coords = parse_xyz_and_eda_from_output!(two_body_output_file, eda_data)
    
    map!(x->Float64[], values(eda_data))
    trimer_labels, trimer_coords = parse_xyz_and_eda_from_output!(three_body_output_file, eda_data)
    map!(x->Float64[], values(eda_data))
    labels, coords = parse_xyz_and_eda_from_output!(full_output_file, eda_data)
    
    # Everything in here assumes there is a single ion and the rest is water and
    # that ion comes at the beginning of the xyz coordinates.
    num_fragments = ((length(labels[1]) - 1) ÷ 3) + 1
    num_dimers = length(dimer_coords)
    num_expected_dimers = binomial(num_fragments, 2)
    num_trimers = length(trimer_coords)
    num_expected_trimers = binomial(num_fragments, 3)
    @assert binomial(num_fragments, 2) == num_dimers "Parsed $num_dimers but expected $num_expected_dimers based on there being $num_fragments fragments."
    @assert binomial(num_fragments, 3) == num_trimers "Parsed $num_trimers but expected $num_expected_trimers based on there being $num_fragments fragments."

    num_ion_water_dimers    = binomial(num_fragments-1, 1)
    num_ion_water_trimers   = binomial(num_fragments-1, 2)

    two_body_eda_data = Dict(
        :cls_elec => Float64[],
        :mod_pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[],
    )

    three_body_eda_data = Dict(
        :cls_elec => Float64[],
        :mod_pauli => Float64[],
        :disp => Float64[],
        :pol => Float64[],
        :ct => Float64[],
    )

    # Parse two and three body contributions to EDA MBE #
    parse_EDA_terms!(two_body_eda_data, two_body_output_file, false)
    parse_EDA_terms!(three_body_eda_data, three_body_output_file, false)

    # make map of fragment indices to linear dimer index and linear trimer index
    dimer_map  = Dict{Tuple{Int, Int}, Int}()
    trimer_map = Dict{Tuple{Int, Int, Int}, Int}()
    i_linear = 1
    for i in 1:num_fragments-1
        for j in (i+1):num_fragments
            dimer_map[(i, j)] = i_linear
            i_linear += 1
        end
    end
    i_linear = 1
    for i in 1:num_fragments-2
        for j in (i+1):num_fragments-1
            for k in (j+1):num_fragments
                trimer_map[(i, j, k)] = i_linear
                i_linear += 1
            end
        end
    end
    
    # add 3-body keys to trimer dict #
    for key in keys(two_body_eda_data) # because we don't add keys to two_body_eda_data and it has the same original keys as three_body_eda_dict
        three_body_eda_data[Symbol(key, :_3b)] = zeros(length(three_body_eda_data[key]))
    end
    # Get non-additive contributions to each trimer #
    for i in 1:num_fragments-2
        for j in (i+1):num_fragments-1
            for k in (j+1):num_fragments
                i_trimer = trimer_map[(i, j, k)]
                i_dimer_1 = dimer_map[(i, j)]
                i_dimer_2 = dimer_map[(i, k)]
                i_dimer_3 = dimer_map[(j, k)]
                for key in keys(two_body_eda_data)
                    e_3b = three_body_eda_data[key][i_trimer] - two_body_eda_data[key][i_dimer_1] - two_body_eda_data[key][i_dimer_2] - two_body_eda_data[key][i_dimer_3]
                    three_body_eda_data[Symbol(key, :_3b)][i_trimer] = e_3b
                end
            end
        end
    end
    two_body_eda_data[:has_ion]   = [i <= num_ion_water_dimers ? true : false for i in 1:num_dimers]
    three_body_eda_data[:has_ion] = [i <= num_ion_water_trimers ? true : false for i in 1:num_trimers]

    # get dimer, trimer, and full coordinates in format to return
    dimer_coords_out = [[MVector{3, Float64}(dimer_coords[i_geom][:, i]) for i in eachindex(eachcol(dimer_coords[i_geom]))] for i_geom in eachindex(dimer_coords)]
    trimer_coords_out = [[MVector{3, Float64}(trimer_coords[i_geom][:, i]) for i in eachindex(eachcol(trimer_coords[i_geom]))] for i_geom in eachindex(trimer_coords)]
    full_system_labels = labels[1]
    full_system_coords = [MVector{3, Float64}(coords[1][:, i]) for i in eachindex(eachcol(coords[1]))]

    return dimer_labels, dimer_coords_out, trimer_labels, trimer_coords_out, full_system_labels, full_system_coords, two_body_eda_data, three_body_eda_data, eda_data
end

function parse_mbe_eda_ion_water_data_and_write_to_csv_2_body_many_body(csv_outfile::String, xyz_outfile::String)

    all_files = readdir()
    full_system_files = Int[]
    two_body_files = Int[]
    for i in eachindex(all_files)
        if occursin("full_system", all_files[i])
            push!(full_system_files, i)
        end
        if occursin("2_body", all_files[i])
            push!(two_body_files, i)
        end
    end

    file_pairs = Tuple{Int, Int}[]
    for i_full in full_system_files
        full_system_file = all_files[i_full]
        two_body_file_index = 0

        two_body_file_prefix = split(full_system_file, "full_system.out")[1]
        for i_two_body in two_body_files
            if occursin(two_body_file_prefix, all_files[i_two_body])
                two_body_file_index = i_two_body
                break
            end
        end
        if (two_body_file_index > 0)
            push!(file_pairs, (i_full, two_body_file_index))
        end
    end

    # Now that we've found all the full_system, 2_body, 3_body triples,
    # let's parse the structures, and EDA data and write it to a CSV
    # for further analysis.
    all_pauli_iw  = zeros(length(file_pairs))
    all_pauli_ww  = zeros(length(file_pairs))
    all_elec_iw  = zeros(length(file_pairs))
    all_elec_ww  = zeros(length(file_pairs))
    all_disp_iw  = zeros(length(file_pairs))
    all_disp_ww  = zeros(length(file_pairs))
    all_pol_iw  = zeros(length(file_pairs))
    all_pol_ww  = zeros(length(file_pairs))
    all_ct_iw  = zeros(length(file_pairs))
    all_ct_ww  = zeros(length(file_pairs))

    all_pauli_mb = zeros(length(file_pairs))
    all_elec_mb = zeros(length(file_pairs))
    all_disp_mb = zeros(length(file_pairs))
    all_pol_mb = zeros(length(file_pairs))
    all_ct_mb = zeros(length(file_pairs))
    all_pauli_total = zeros(length(file_pairs))
    all_elec_total = zeros(length(file_pairs))
    all_disp_total = zeros(length(file_pairs))
    all_pol_total = zeros(length(file_pairs))
    all_ct_total = zeros(length(file_pairs))

    all_geoms = Matrix{Float64}[]
    all_labels = Vector{String}[]
    all_ion_labels = String[]
    all_num_waters = zeros(length(file_pairs))
    for i in ProgressBar(eachindex(file_pairs))
        full_labels, full_geom, mbe_eda = process_EDA_mbe_ion_water_calculation(
            all_files[file_pairs[i][1]],
            all_files[file_pairs[i][2]]
        )
        push!(all_ion_labels, full_labels[1])
        push!(all_geoms, reduce(hcat, full_geom))
        push!(all_labels, full_labels)
        all_pauli_iw[i]    = mbe_eda[:mod_pauli][1]
        all_pauli_ww[i]    = mbe_eda[:mod_pauli][2]
        all_pauli_mb[i]   = mbe_eda[:mod_pauli][3]
        all_pauli_total[i] = mbe_eda[:mod_pauli][4]

        all_elec_iw[i]    = mbe_eda[:cls_elec][1]
        all_elec_ww[i]    = mbe_eda[:cls_elec][2]
        all_elec_mb[i]   = mbe_eda[:cls_elec][3]
        all_elec_total[i] = mbe_eda[:cls_elec][4]

        all_disp_iw[i]    = mbe_eda[:disp][1]
        all_disp_ww[i]    = mbe_eda[:disp][2]
        all_disp_mb[i]   = mbe_eda[:disp][3]
        all_disp_total[i] = mbe_eda[:disp][4]

        all_pol_iw[i]    = mbe_eda[:pol][1]
        all_pol_ww[i]    = mbe_eda[:pol][2]
        all_pol_mb[i]   = mbe_eda[:pol][3]
        all_pol_total[i] = mbe_eda[:pol][4]

        all_ct_iw[i]    = mbe_eda[:ct][1]
        all_ct_ww[i]    = mbe_eda[:ct][2]
        all_ct_mb[i]   = mbe_eda[:ct][3]
        all_ct_total[i] = mbe_eda[:ct][4]

        all_num_waters[i] = (length(full_labels) - 1) / 3
    end

    final_data = Dict(
        :pauli_iw => all_pauli_iw,
        :pauli_ww => all_pauli_ww,
        :pauli_mb => all_pauli_mb,
        :pauli => all_pauli_total,
        :elec_iw => all_elec_iw,
        :elec_ww => all_elec_ww,
        :elec_mb => all_elec_mb,
        :elec => all_elec_total,
        :disp_iw => all_disp_iw,
        :disp_ww => all_disp_ww,
        :disp_mb => all_disp_mb,
        :disp => all_disp_total,
        :pol_iw => all_pol_iw,
        :pol_ww => all_pol_ww,
        :pol_mb => all_pol_mb,
        :pol => all_pol_total,
        :ct_iw => all_ct_iw,
        :ct_ww => all_ct_ww,
        :ct_mb => all_ct_mb,
        :ct => all_ct_total,
        :num_waters => all_num_waters,
        :ion_labels => all_ion_labels
    )

    df = DataFrame(final_data)
    geom_index = [1:nrow(df)...]
    df[!, :index] = geom_index
    write_xyz(xyz_outfile, [string(length(all_labels[i]), "\n") for i in eachindex(all_labels)], all_labels, all_geoms)
    CSV.write(csv_outfile, df)
    return
end

function parse_mbe_eda_ion_water_data_and_write_to_csv(csv_outfile::String, xyz_outfile::String)

    all_files = readdir()
    full_system_files = Int[]
    two_body_files = Int[]
    three_body_files = Int[]
    for i in eachindex(all_files)
        if occursin("full_system", all_files[i])
            push!(full_system_files, i)
        end
        if occursin("2_body", all_files[i])
            push!(two_body_files, i)
        end
        if occursin("3_body", all_files[i])
            push!(three_body_files, i)
        end
    end

    file_triples = Tuple{Int, Int, Int}[]
    for i_full in full_system_files
        full_system_file = all_files[i_full]
        two_body_file_index = 0
        three_body_file_index = 0

        two_body_file_prefix = split(full_system_file, "full_system.out")[1]
        three_body_file_prefix = split(full_system_file, "full_system.out")[1]
        for i_two_body in two_body_files
            if occursin(two_body_file_prefix, all_files[i_two_body])
                two_body_file_index = i_two_body
                break
            end
        end
        for i_three_body in three_body_files
            if occursin(three_body_file_prefix, all_files[i_three_body])
                three_body_file_index = i_three_body
                break
            end
        end
        if (two_body_file_index > 0 && three_body_file_index > 0)
            push!(file_triples, (i_full, two_body_file_index, three_body_file_index))
        end
    end

    # Now that we've found all the full_system, 2_body, 3_body triples,
    # let's parse the structures, and EDA data and write it to a CSV
    # for further analysis.
    all_pauli_iw  = zeros(length(file_triples))
    all_pauli_iww = zeros(length(file_triples))
    all_pauli_ww  = zeros(length(file_triples))
    all_pauli_www = zeros(length(file_triples))
    all_elec_iw  = zeros(length(file_triples))
    all_elec_iww = zeros(length(file_triples))
    all_elec_ww  = zeros(length(file_triples))
    all_elec_www = zeros(length(file_triples))
    all_disp_iw  = zeros(length(file_triples))
    all_disp_iww = zeros(length(file_triples))
    all_disp_ww  = zeros(length(file_triples))
    all_disp_www = zeros(length(file_triples))
    all_pol_iw  = zeros(length(file_triples))
    all_pol_iww = zeros(length(file_triples))
    all_pol_ww  = zeros(length(file_triples))
    all_pol_www = zeros(length(file_triples))
    all_ct_iw  = zeros(length(file_triples))
    all_ct_iww = zeros(length(file_triples))
    all_ct_ww  = zeros(length(file_triples))
    all_ct_www = zeros(length(file_triples))

    all_pauli_total = zeros(length(file_triples))
    all_elec_total = zeros(length(file_triples))
    all_disp_total = zeros(length(file_triples))
    all_pol_total = zeros(length(file_triples))
    all_ct_total = zeros(length(file_triples))

    all_geoms = Matrix{Float64}[]
    all_labels = Vector{String}[]
    all_ion_labels = String[]
    all_num_waters = zeros(length(file_triples))
    for i in ProgressBar(eachindex(file_triples))
        full_labels, full_geom, mbe_eda = process_EDA_mbe_ion_water_calculation(
            all_files[file_triples[i][1]],
            all_files[file_triples[i][2]],
            all_files[file_triples[i][3]]
        )
        push!(all_ion_labels, full_labels[1])
        push!(all_geoms, reduce(hcat, full_geom))
        push!(all_labels, full_labels)
        all_pauli_iw[i]    = mbe_eda[:mod_pauli][1]
        all_pauli_ww[i]    = mbe_eda[:mod_pauli][2]
        all_pauli_iww[i]   = mbe_eda[:mod_pauli][3]
        all_pauli_www[i]   = mbe_eda[:mod_pauli][4]
        all_pauli_total[i] = mbe_eda[:mod_pauli][5]

        all_elec_iw[i]    = mbe_eda[:cls_elec][1]
        all_elec_ww[i]    = mbe_eda[:cls_elec][2]
        all_elec_iww[i]   = mbe_eda[:cls_elec][3]
        all_elec_www[i]   = mbe_eda[:cls_elec][4]
        all_elec_total[i] = mbe_eda[:cls_elec][5]

        all_disp_iw[i]    = mbe_eda[:disp][1]
        all_disp_ww[i]    = mbe_eda[:disp][2]
        all_disp_iww[i]   = mbe_eda[:disp][3]
        all_disp_www[i]   = mbe_eda[:disp][4]
        all_disp_total[i] = mbe_eda[:disp][5]

        all_pol_iw[i]    = mbe_eda[:pol][1]
        all_pol_ww[i]    = mbe_eda[:pol][2]
        all_pol_iww[i]   = mbe_eda[:pol][3]
        all_pol_www[i]   = mbe_eda[:pol][4]
        all_pol_total[i] = mbe_eda[:pol][5]

        all_ct_iw[i]    = mbe_eda[:ct][1]
        all_ct_ww[i]    = mbe_eda[:ct][2]
        all_ct_iww[i]   = mbe_eda[:ct][3]
        all_ct_www[i]   = mbe_eda[:ct][4]
        all_ct_total[i] = mbe_eda[:ct][5]

        all_num_waters[i] = (length(full_labels) - 1) / 3
    end

    final_data = Dict(
        :pauli_iw => all_pauli_iw,
        :pauli_ww => all_pauli_ww,
        :pauli_iww => all_pauli_iww,
        :pauli_www => all_pauli_www,
        :pauli => all_pauli_total,
        :elec_iw => all_elec_iw,
        :elec_ww => all_elec_ww,
        :elec_iww => all_elec_iww,
        :elec_www => all_elec_www,
        :elec => all_elec_total,
        :disp_iw => all_disp_iw,
        :disp_ww => all_disp_ww,
        :disp_iww => all_disp_iww,
        :disp_www => all_disp_www,
        :disp => all_disp_total,
        :pol_iw => all_pol_iw,
        :pol_ww => all_pol_ww,
        :pol_iww => all_pol_iww,
        :pol_www => all_pol_www,
        :pol => all_pol_total,
        :ct_iw => all_ct_iw,
        :ct_ww => all_ct_ww,
        :ct_iww => all_ct_iww,
        :ct_www => all_ct_www,
        :ct => all_ct_total,
        :num_waters => all_num_waters,
        :ion_labels => all_ion_labels
    )

    df = DataFrame(final_data)
    geom_index = [1:nrow(df)...]
    df[!, :index] = geom_index
    write_xyz(xyz_outfile, [string(length(all_labels[i]), "\n") for i in eachindex(all_labels)], all_labels, all_geoms)
    CSV.write(csv_outfile, df)
    return
end

function parse_mbe_eda_ion_water_data_without_combining_and_write_to_csv(
    csv_outfile_dimers::String, xyz_outfile_dimers::String,
    csv_outfile_trimers::String, xyz_outfile_trimers::String,
    mandatory_substring::Union{String, Nothing}=nothing
)

    all_files = readdir()
    full_system_files = Int[]
    two_body_files = Int[]
    three_body_files = Int[]
    for i in eachindex(all_files)
        if mandatory_substring === nothing || occursin(mandatory_substring, all_files[i])
            if occursin("full_system", all_files[i])
                push!(full_system_files, i)
            end
            if occursin("2_body", all_files[i])
                push!(two_body_files, i)
            end
            if occursin("3_body", all_files[i])
                push!(three_body_files, i)
            end
        end
    end

    file_triples = Tuple{Int, Int, Int}[]
    for i_full in full_system_files
        full_system_file = all_files[i_full]
        two_body_file_index = 0
        three_body_file_index = 0

        two_body_file_prefix = split(full_system_file, "_full_system.out")[1]
        three_body_file_prefix = split(full_system_file, "_full_system.out")[1]
        for i_two_body in two_body_files
            if occursin(two_body_file_prefix, all_files[i_two_body])
                two_body_file_index = i_two_body
                break
            end
        end
        for i_three_body in three_body_files
            if occursin(three_body_file_prefix, all_files[i_three_body])
                three_body_file_index = i_three_body
                break
            end
        end
        if (two_body_file_index > 0 && three_body_file_index > 0)
            push!(file_triples, (i_full, two_body_file_index, three_body_file_index))
        end
    end

    all_pauli_dimers = Float64[]
    all_disp_dimers = Float64[]
    all_elec_dimers = Float64[]
    all_pol_dimers = Float64[]
    all_ct_dimers = Float64[]
    all_has_ion_dimers = Float64[]

    all_pauli_trimers = Float64[]
    all_pauli_trimers_3b = Float64[]
    all_disp_trimers = Float64[]
    all_disp_trimers_3b = Float64[]
    all_elec_trimers = Float64[]
    all_elec_trimers_3b = Float64[]
    all_pol_trimers = Float64[]
    all_pol_trimers_3b = Float64[]
    all_ct_trimers = Float64[]
    all_ct_trimers_3b = Float64[]
    all_has_ion_trimers = Float64[]

    all_dimer_geoms = Matrix{Float64}[]
    all_dimer_labels = Vector{String}[]
    all_trimer_geoms = Matrix{Float64}[]
    all_trimer_labels = Vector{String}[]
    for i in ProgressBar(eachindex(file_triples))
        # ...... That's a lot of things ...... #
        dimer_labels, dimer_coords_out, trimer_labels, trimer_coords_out,
        _, _, two_body_eda_data, three_body_eda_data, _ = process_EDA_mbe_ion_water_calculation_into_all_many_body_terms(
            all_files[file_triples[i][1]],
            all_files[file_triples[i][2]],
            all_files[file_triples[i][3]]
        )
        append!(all_dimer_labels, dimer_labels)
        append!(all_dimer_geoms, [reduce(hcat, dimer_geom) for dimer_geom in dimer_coords_out])
        append!(all_trimer_labels, trimer_labels)
        append!(all_trimer_geoms, [reduce(hcat, trimer_geom) for trimer_geom in trimer_coords_out])

        for key in keys(two_body_eda_data)
            if key == :cls_elec
                append!(all_elec_dimers, two_body_eda_data[key])
            elseif key == :mod_pauli
                append!(all_pauli_dimers, two_body_eda_data[key])
            elseif key == :disp
                append!(all_disp_dimers, two_body_eda_data[key])
            elseif key == :pol
                append!(all_pol_dimers, two_body_eda_data[key])
            elseif key == :ct
                append!(all_ct_dimers, two_body_eda_data[key])
            elseif key == :has_ion
                append!(all_has_ion_dimers, two_body_eda_data[key])
            end
        end
        for key in keys(three_body_eda_data)
            if key == :cls_elec
                append!(all_elec_trimers, three_body_eda_data[key])
            elseif key == :mod_pauli
                append!(all_pauli_trimers, three_body_eda_data[key])
            elseif key == :disp
                append!(all_disp_trimers, three_body_eda_data[key])
            elseif key == :pol
                append!(all_pol_trimers, three_body_eda_data[key])
            elseif key == :ct
                append!(all_ct_trimers, three_body_eda_data[key])
            elseif key == :cls_elec_3b
                append!(all_elec_trimers_3b, three_body_eda_data[key])
            elseif key == :mod_pauli_3b
                append!(all_pauli_trimers_3b, three_body_eda_data[key])
            elseif key == :disp_3b
                append!(all_disp_trimers_3b, three_body_eda_data[key])
            elseif key == :pol_3b
                append!(all_pol_trimers_3b, three_body_eda_data[key])
            elseif key == :ct_3b
                append!(all_ct_trimers_3b, three_body_eda_data[key])
            elseif key == :has_ion
                append!(all_has_ion_trimers, three_body_eda_data[key])
            end
        end
    end

    all_dimer_data = Dict(
        :cls_elec => all_elec_dimers,
        :mod_pauli => all_pauli_dimers,
        :disp => all_disp_dimers,
        :pol => all_pol_dimers,
        :ct => all_ct_dimers,
        :int => all_elec_dimers + all_pauli_dimers + all_disp_dimers + all_pol_dimers + all_ct_dimers,
        :has_ion => all_has_ion_dimers
    )

    all_trimer_data = Dict(
        :cls_elec => all_elec_trimers,
        :cls_elec_3b => all_elec_trimers_3b,
        :mod_pauli => all_pauli_trimers,
        :mod_pauli_3b => all_pauli_trimers_3b,
        :disp => all_disp_trimers,
        :disp_3b => all_disp_trimers_3b,
        :pol => all_pol_trimers,
        :pol_3b => all_pol_trimers_3b,
        :ct => all_ct_trimers,
        :ct_3b => all_ct_trimers_3b,
        :int => all_elec_trimers + all_pauli_trimers + all_disp_trimers + all_pol_trimers + all_ct_trimers,
        :int_3b => all_elec_trimers_3b + all_pauli_trimers_3b + all_disp_trimers_3b + all_pol_trimers_3b + all_ct_trimers_3b,
        :has_ion => all_has_ion_trimers
    )

    df_2b = DataFrame(all_dimer_data)
    df_3b = DataFrame(all_trimer_data)
    geom_index = [1:nrow(df_2b)...]
    df_2b[!, :index] = geom_index
    geom_index = [1:nrow(df_3b)...]
    df_3b[!, :index] = geom_index
    write_xyz(xyz_outfile_dimers, [string(length(all_dimer_labels[i]), "\n") for i in eachindex(all_dimer_labels)], all_dimer_labels, all_dimer_geoms)
    CSV.write(csv_outfile_dimers, df_2b)
    write_xyz(xyz_outfile_trimers, [string(length(all_trimer_labels[i]), "\n") for i in eachindex(all_trimer_labels)], all_trimer_labels, all_trimer_geoms)
    CSV.write(csv_outfile_trimers, df_3b)
    return
end
