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