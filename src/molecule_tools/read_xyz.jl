using HybridArrays, StaticArrays

function read_xyz(ifile::String; T::Type=Float64, static::Bool=false, start_at_N::Int=1, load_N_frames::Int=typemax(Int))
    """
    Reads in an xyz file of possibly multiple geometries, returning the header, atom labels, 
    and coordinates as arrays of strings and Float64s for the coordinates.
    If load_N_frames is specified (greater than zero),
	only the first N geometries will be read starting from start_at_N.
    Otherwise, all geoms will be read.
    """
    header = Vector{String}()
    atom_labels = Vector{Vector{String}}()
    if static
        geoms = Vector{HybridArray{Tuple{3,StaticArrays.Dynamic()}}}()
    else
        geoms = Vector{Matrix{T}}()
    end
	i_frame = 1
	successfully_parsed = true
    open(ifile, "r") do io
        for line in eachline(io)
            if isa(tryparse(Int, line), Int)
				successfully_parsed = true
                # allocate the geometry for this frame
                N = parse(Int, line)
				if (i_frame >= start_at_N)

					# store the header for this frame
					head = string(line, "\n", readline(io))
					# loop through the geometry storing the vectors and atom labels as you go
					new_data = fill(Vector{SubString{String}}(), N)
					geom = zeros((3,N))
					labels = fill("", N)

					for j = 1:N
						new_data[j] = split(readline(io))
					end
					for j = 1:N
						try
							labels[j] = new_data[j][1]
							geom[1,j] = parse(T, new_data[j][2])
							geom[2,j] = parse(T, new_data[j][3])
							geom[3,j] = parse(T, new_data[j][4])
						catch
							@warn "Failed to parse on frame " i_frame j new_data[j] "Skipping this frame and moving on."
							successfully_parsed = false
						end
					end
					if successfully_parsed
						push!(header, head)
						push!(geoms, geom)
						push!(atom_labels, labels)
					end
				else
					# dump the frame since we are skipping to start_at_N
					for _ in 1:N
						readline(io)
					end
				end
				i_frame += 1
            end
            if length(header) == load_N_frames
                break
            end
            if (length(header) % 500) == 0 && length(header) > 1
                println(string("Finished reading ", length(header), " structures."))
            end
        end
    end

    return header, atom_labels, geoms
end

function read_fragmented_xyz(ifile::String; T::Type=Float64)
    """
    Reads in an xyz file with one fragmented geometry, returning the header, atom labels, 
    and coordinates as arrays of strings and Float64s for the coordinates.
    The fragments are separated by '--' in each case.
    """

    header = Array{String, 1}()
    atom_labels = Array{Array{String, 1}, 1}()
    geoms = Vector{Matrix{T}}()
    open(ifile, "r") do io
        line = readline(io)
        # allocate the geometry for this frame
        N = parse(Int, line)

        # store the header for this frame
        head = string(line, "\n", readline(io))
        push!(header, head)
        # loop through the geometry storing the vectors and atom labels as you go
        new_geom::Vector{Vector{T}} = []
        labels::Vector{String} = []
        while !eof(io) 
            line = readline(io)
            if line == "--"
                push!(geoms, hcat(new_geom...))
                push!(atom_labels, labels)
                new_geom = []
                labels = []
                continue
            end

            split_line = split(line)

            push!(labels, split_line[1])
            push!(new_geom, [parse(T, split_line[2]), parse(T, split_line[3]), parse(T, split_line[4])])
        end
        push!(geoms, hcat(new_geom...))
        push!(atom_labels, labels)
    end
    return repeat(header, length(atom_labels)), atom_labels, geoms
end

function write_xyz(outfile::AbstractString, header::AbstractArray, labels::AbstractArray, geoms::AbstractArray; append::Bool=false, directory::AbstractString="")
    """
    Writes an xyz file with the elements returned by read_xyz.
    """
    if isdir(directory)
        outfile = string(directory, "/", outfile)
    elseif directory == ""
        
    else
        mkdir(directory)
        outfile = string(directory, "/", outfile)
    end
    mode = "w"
    if append
        mode = "a"
    end
    if length(header) != length(geoms)
        header = [header[1] for i in 1:length(geoms)]
    end
    if length(labels) != length(geoms)
        labels = [labels[1] for i in 1:length(geoms)]
    end
    open(outfile, mode) do io
        for (i_geom, head) in enumerate(header)
            write(io, string(head, "\n"))
            for (i_coord, atom_label) in enumerate(labels[i_geom])
                write(io, string(atom_label, " ", join(string.(geoms[i_geom][:,i_coord]), " "), "\n"))
            end
        end
    end
end

function write_xyz(outfile::AbstractString, header::AbstractString, labels::AbstractVector{String}, geoms::AbstractVector{Matrix{Float64}}; append::Bool=false, directory::AbstractString="")
    write_xyz(outfile, [header for _ in eachindex(geoms)], [labels for _ in eachindex(geoms)], geoms, append=append, directory=directory)
end

function write_xyz(outfile::AbstractString, labels::AbstractVector{String}, geom::Matrix{Float64}; append::Bool=false, directory::AbstractString="")
    write_xyz(outfile, [string(length(labels), "\n")], [labels], [geom], append=append, directory=directory)
end

function write_xyz(outfile::AbstractString, labels::AbstractVector{Vector{String}}, geoms::AbstractVector{Matrix{Float64}}; append::Bool=false, directory::AbstractString="")
    write_xyz(outfile, [string(length(labels[i]), "\n") for i in eachindex(labels)], labels, geoms, append=append, directory=directory)
end