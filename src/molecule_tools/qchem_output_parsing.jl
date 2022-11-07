using LinearAlgebra

"""
Parses the wiberg bond order matrix from a Q-Chem output file.
Returns an array of all bond order matrices found in the file.
"""
function parse_wiberg_BO_matrix(output_file::String)
    file_contents = readlines(output_file)
    BO_matrices = Matrix{Float64}[]
    for (i, line) in enumerate(file_contents)
        if occursin("Wiberg bond index matrix in the NAO basis:", line)
            line_idx = i+4 # takes you to first line of matrix
            
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
            line_idx = i+4
            for i_block in 1:num_BO_blocks
                for i_atom in 1:natoms
                    BO_line = file_contents[line_idx]
                    split_line = split(BO_line)[3:end]
                    BOs = tryparse.((Float64,), split_line)
                    for j in eachindex(BOs)
                        # 9 is the number of atoms printed per block
                        BO_matrix[i_atom, (i_block-1) * 9 + j] = BOs[j]
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
    cls_elec  = Float64[]
    elec      = Float64[]
    mod_pauli = Float64[]
    pauli     = Float64[]
    disp      = Float64[]
    pol       = Float64[]
    ct        = Float64[]

    lines = readlines(output_file)
    for line in lines
        if occursin("(ELEC)", line)
            push!(elec, tryparse(Float64, split(line)[5]))
        end
        if occursin("(PAULI)", line)
            push!(pauli, tryparse(Float64, split(line)[5]))
        end
        if occursin("   (DISP)   ", line)
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
        :cls_elec  => cls_elec,
        :elec      => elec,
        :mod_pauli => mod_pauli,
        :pauli     => pauli,
        :disp      => disp,
        :pol       => pol,
        :ct        => ct
    )
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