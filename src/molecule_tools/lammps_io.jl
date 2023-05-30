using ProgressMeter

function parse_lammps_trajectory(infile::String, type_to_label::Union{Dict{Int, String}, Nothing}=nothing)
    coords = Matrix{Float64}[]
    atom_types = Vector{Int}[]

    lines = readlines(infile)
    natoms = 0
    parse_geom = false
    @showprogress for (i, line) in enumerate(lines)
        @label skip_frame
        if occursin("ITEM: NUMBER OF ATOMS", line)
            natoms = tryparse(Int, lines[i+1])
            push!(atom_types, zeros(Int, natoms))
            push!(coords, zeros(3, natoms))
            continue
        end
        if occursin("ITEM: ATOMS", line)
            parse_geom = true
            continue
        end
        if parse_geom
            for j in 1:natoms
                split_line = split(lines[i-1+j])
                if length(split_line) != 5
                    @warn "Failed to parse a frame in lammps file. Skipping and moving to next frame."
                    i += 1
                    pop!(atom_types)
                    pop!(coords)
                    parse_geom = false
                    @goto skip_frame
                end
                atom_type = tryparse(Int, split_line[1])
                new_coords = [tryparse(Float64, split_line[3]), tryparse(Float64, split_line[4]), tryparse(Float64, split_line[5])]
                if atom_type !== nothing
                    atom_types[end][j] = atom_type
                else
                    @warn "Failed to parse a frame in lammps file. Skipping and moving to next frame."
                    i += 1
                    pop!(atom_types)
                    pop!(coords)
                    parse_geom = false
                    @goto skip_frame
                end
                if !any(==(true), isnothing.(new_coords))
                    @views coords[end][:, j] = new_coords
                else
                    @warn "Failed to parse a frame in lammps file. Skipping and moving to next frame."
                    i += 1
                    pop!(atom_types)
                    pop!(coords)
                    parse_geom = false
                    @goto skip_frame
                end

            end
            parse_geom = false
        end
    end
    if type_to_label !== nothing
        atom_labels = [[type_to_label[atom_types[i][j]] for j in eachindex(atom_types[i])] for i in eachindex(atom_types)]
        return atom_labels, coords
    end
    return atom_types, coords
end

function generate_reaxff_cgem_configuration_with_shells(coords::Matrix{Float64}, labels::Vector{String}, add_extra_shell::Vector{Int}=Int[], no_shell::Vector{Int}=Int[], label_to_type::Dict{String, Int}=Dict("O"=>2, "H"=>1))
    coords_with_shells = Vector{Float64}[]
    charges = Int[]
    types = Int[]
    for i in eachindex(labels)
        push!(coords_with_shells, coords[:,i])
        push!(charges, 1)
        push!(types, label_to_type[labels[i]])
        if i in no_shell
            continue
        end
        push!(coords_with_shells, coords[:,i] + randn(3) * 0.01)
        push!(charges, -1)
        push!(types, 3) # shell type is always 3
        if i in add_extra_shell
            push!(coords_with_shells, coords[:,i] + randn(3) * 0.01)
            push!(charges, -1)
            push!(types, 3) # shell type is always 3
        end
    end

    open("cgem_initial_structure.init", "w") do io
        for i in eachindex(types)
            write(io, string(i, " ", types[i], " ", charges[i], " ", coords_with_shells[i][1], " ", coords_with_shells[i][2], " ", coords_with_shells[i][3], "\n"))
        end
    end
end