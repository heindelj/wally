using HybridArrays, StaticArrays

function read_xyz(ifile::String; T::Type=Float64, static::Bool=false, up_to_N::Int=0)
    """
    Reads in an xyz file of possibly multiple geometries, returning the header, atom labels, 
    and coordinates as arrays of strings and Float64s for the coordinates.
    If up_to_N is specified (greater than zero), only the first N geometries will be read.
    Otherwise, all geoms will be read.
    """
    if up_to_N == 0
        up_to_N = Integer(maxintfloat(Float64)) # a large number
    end

    header = Array{String, 1}()
    atom_labels = Array{Array{String, 1}, 1}()
    if static
        geoms = Array{HybridArray{Tuple{3,StaticArrays.Dynamic()}}, 1}()
    else
        geoms = Array{Array{T, 2}, 1}()
    end
    open(ifile, "r") do io
        for line in eachline(io)
            if isa(tryparse(Int, line), Int)
                # allocate the geometry for this frame
                N = parse(Int, line)

                # store the header for this frame
                head = string(line, "\n", readline(io))
                push!(header, head)
                # loop through the geometry storing the vectors and atom labels as you go
                new_data = fill(Array{SubString{String},1}(), N)
                geom = zeros((3,N))
                labels = fill("", N)

                @inbounds for j = 1:N
                    new_data[j] = split(readline(io))
                end

                @inbounds for j = 1:N
                    labels[j] = new_data[j][1]
                    geom[1,j] = parse(T, new_data[j][2])
                    geom[2,j] = parse(T, new_data[j][3])
                    geom[3,j] = parse(T, new_data[j][4])
                end
                push!(geoms, geom)
                push!(atom_labels, labels)
            end
            if length(header) == up_to_N
                break
            end
        end
    end

    return header, atom_labels, geoms
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