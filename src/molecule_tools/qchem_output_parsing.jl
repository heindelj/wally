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
                push!(eda_dict[:deform], (fragment_sum - num_fragments * fragment_zero) * 627.51 * 4.184)
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
        :total => Float64[]
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
        parse_EDA_terms!(mbe_eda_data, mbe_file, true, E_h2o)
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
    parse_EDA_terms!(total_eda_data, full_output_file, true, E_h2o)
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
    for i in eachindex(total_eda_data[:int])
        if i < length(total_eda_data[:int])
            total_eda_data[:deform][i] = total_eda_data[:deform][end]
        end
        push!(total_eda_data[:total], total_eda_data[:int][i] + total_eda_data[:deform][end])
        # ^^ Always add the end of deformation since the MBE one gets summed too many times
        # and the deformation at an MBE level and full level are always the same.
    end

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

