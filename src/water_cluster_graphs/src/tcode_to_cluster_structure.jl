using ProgressBars

include("../../molecule_tools/water_tools.jl")
include("../../molecule_tools/read_xyz.jl")
include("../../molecule_tools/molecular_axes.jl") # centroid
include("../../molecule_tools/optimize_xyz.jl")
using Base.Filesystem


mutable struct tcode
    indices::Vector{Int}
end

function load_tcode_file(file_name::AbstractString)
    """
    Takes a file containing (possibly) many directed graphs represented via a tcode 
    and loads them into an array of tcode structs.
    Note that tcodes are typicall 0-indexed so I elevate them to 1 indexed here for convenience.
    """
    all_lines = readlines(file_name)
    tcodes = Array{tcode, 1}(undef, length(all_lines))
    Threads.@threads for i in 1:length(all_lines)
        tcodes[i] = tcode(parse.(Int, split(all_lines[i])))
        for j in 3:length(tcodes[i].indices)
            tcodes[i].indices[j] += 1
        end
    end
    return tcodes
end

function get_hbond_partners_from_tcode(tcode_array::tcode)
    """
    Takes a tcode and splits it into tuples which represent the donor and acceptor
    oxygen atoms, respectively.
    """
    @assert isinteger(length(tcode_array.indices) / 2) "tcode doesn't have even number of indices."
    pairs::Array{Tuple,1} = []
    for i in 3:2:length(tcode_array.indices)
        push!(pairs, (tcode_array.indices[i], tcode_array.indices[i+1]))
    end
    return pairs
end

function angle_between_vectors(a::AbstractVector, b::AbstractVector)
    return acos(clamp(a ⋅ b / (norm(a) * norm(b)), -1.0, 1.0))
end

function rotate_about_axis(v::AbstractVector, axis::AbstractVector, angle::T) where T <: Real
    """
    Rotates vec about axis by the given angle (in radians), in a ccw fashion.
    """
    axis /= norm(axis)
    a = cos(angle / 2.0)
    b, c, d = -axis * sin(angle / 2.0)
    aa, bb, cc, dd = (a * a, b * b, c * c, d * d)
    bc, ad, ac, ab, bd, cd = (b * c, a * d, a * c, a * b, b * d, c * d)
    rot::Array{Float64,2} = [[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)] [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)] [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]
    return rot' * v
end

function dangling_hydrogen_from_centroid(vec_O::AbstractVector, centroid::AbstractVector; OH_distance::Float64 = 0.9572)
    line = vec_O - centroid
    line = line * (norm(line) + OH_distance) / norm(line)
    return centroid + line
end

function structure_from_tcode(t_code::tcode, ref_coords::AbstractMatrix; OH_distance::Float64 = 0.96)
    """
    Takes a tcode and set of reference coordinates for water molecules in OHH order.
    Returns a complete water cluster structure.
    """
    # get the oxygen atom positions at the approproate indices
    remaining_indices = collect(1:size(ref_coords)[2])
    used_indices = collect(1:3:size(ref_coords)[2]) # start with the oxygen indices
    structure = zero(ref_coords)
    structure[:, used_indices] = ref_coords[:, used_indices]

    # place the hydrogen-bonded h atoms
    pairs = get_hbond_partners_from_tcode(t_code)
    for (i, pair) in enumerate(pairs)
        @views h_vec = structure[:, 3 * pair[2] - 2] - structure[:, 3 * pair[1] - 2]
        t = OH_distance / norm(h_vec)
        if iszero(@view(structure[:, (3 * pair[1] - 2) + 1]))
            structure[:, (3 * pair[1] - 2) + 1] = structure[:, 3 * pair[1] - 2] + t * h_vec
            push!(used_indices, (3 * pair[1] - 2) + 1)
        else
            structure[:, (3 * pair[1] - 2) + 2] = structure[:, 3 * pair[1] - 2] + t * h_vec
            push!(used_indices, (3 * pair[1] - 2) + 2)
        end
    end
    # get the indices of free OH atoms
    setdiff!(remaining_indices, used_indices)

    # place the free OH atoms (I slightly randomize so we don't get colinear vectors)
    for i_free_OH in remaining_indices
        @views structure[:, i_free_OH] = dangling_hydrogen_from_centroid(structure[:, 3 * ((i_free_OH - 1) ÷ 3) + 1], vec(centroid(ref_coords))) + rand(3) * 0.0005
    end

    # make all HOH angles 104.5 and distances 0.96
    for i_Oxygen in 1:3:size(ref_coords)[2]
        @views OH_1 = structure[:,i_Oxygen + 1] -  structure[:,i_Oxygen]
        @views OH_2 = structure[:,i_Oxygen + 2] -  structure[:,i_Oxygen]
        bisector = norm(OH_2) * OH_1 + norm(OH_1) * OH_2
        perp = cross(OH_1, OH_2)
        perp /= norm(perp)

        current_angle = angle_between_vectors(OH_1, OH_2)
        deviation = current_angle - 104.5 * pi / 180.0
        
        # rotate the double acceptor OH's and double donor OH's equally
        if i_Oxygen + 1 in remaining_indices
            OH_1 = rotate_about_axis(OH_1, perp,  deviation / 2.0)
            OH_2 = rotate_about_axis(OH_2, perp, -deviation / 2.0)
        elseif i_Oxygen + 1 in used_indices && i_Oxygen + 2 in used_indices
            OH_1 = rotate_about_axis(OH_1, perp,  deviation / 2.0)
            OH_2 = rotate_about_axis(OH_2, perp, -deviation / 2.0)
        else # these are single donors
            OH_2 = rotate_about_axis(OH_2, perp, -deviation)
        end

        @views structure[:,i_Oxygen + 1] = structure[:,i_Oxygen] + OH_1
        @views structure[:,i_Oxygen + 2] = structure[:,i_Oxygen] + OH_2

    end
    return structure
end

function structures_from_tcode(t_codes::AbstractArray{tcode, 1}, ref_coords::AbstractMatrix; OH_distance::Float64 = 0.96)
    """
    Make many structures from the relevant tcodes and reference geometry.
    """
    return structure_from_tcode.(t_codes, (ref_coords,), OH_distance=OH_distance)
end

function optimize_directed_graph_guesses(guess_geoms::AbstractArray{Array{Float64, 2}, 1}, label::AbstractArray{Array{String, 1}}, potential::AbstractPotential; out_file_name::AbstractString="optimized_structures.xyz", write_every::Int=100)
    """
    Optimize the actual guess structures and write the results to specified
    output file.
    """
    for i in ProgressBar(1:(length(guess_geoms) ÷ write_every + 1))
        if i < (length(guess_geoms) ÷ write_every + 1)
            energies, opt_geoms = optimize_structures(@view(guess_geoms[((i-1) * write_every + 1):(i * write_every)]), potential)
        else
            energies, opt_geoms = optimize_structures(@view(guess_geoms[((i-1) * write_every + 1):length(guess_geoms)]), potential)
        end

        labels = []
        headers = []
        for j in 1:length(energies)
            push!(labels, label[1])
            push!(headers, string(size(opt_geoms[j], 2), "\n", energies[j]))
        end

        if i == 1
            write_xyz(out_file_name, headers, labels, opt_geoms)
        else
            write_xyz(out_file_name, headers, labels, opt_geoms, append=true)
        end
    end
end

function optimize_directed_graph_guesses(tcode_file::AbstractString, ref_structure_file::AbstractString, potential::AbstractPotential; out_file_name::AbstractString="optimized_structures.xyz", write_every::Int=100)
    """
    Convenience overload so the potential itself can be passed.
    """

    println("Reading in the tcodes and reference structure...")
    tcodes = load_tcode_file(tcode_file)
    header, label, ref_geom = read_xyz(ref_structure_file)

    println("Forming the guess structures...")
    guess_geoms = structures_from_tcode(tcodes, ref_geom[1])

    println("Doing the optimizations...")

    optimize_directed_graph_guesses(guess_geoms, label, potential, out_file_name=out_file_name)
end

function optimize_directed_graph_guesses(tcode_file::AbstractString, ref_structure_file::AbstractString, potential_constructor::Function; num_tasks::Int=8, out_file_name::AbstractString="optimized_structures.xyz", write_every::Int=100)
    """
    Calls the overload of this function which does the optimization and writing
    but with the optimizations to be split into num_tasks different asynchronous tasks.
    Additionally, rather than taking an AbstractPotential, we take a constructor for
    that potential to avoid a lack of thread safety in the potential being called.
    """
    @assert typeof(potential_constructor()) <: AbstractPotential "potential_constructor needs to construct an AbstractPotential."
                

    println("Reading in the tcodes and reference structure...")
    tcodes = load_tcode_file(tcode_file)
    header, label, ref_geom = read_xyz(ref_structure_file)

    println("Forming the guess structures...")
    guess_geoms = structures_from_tcode(tcodes, ref_geom[1])

    ranges = []
    # get the ranges over which each task will operate
    for i in 1:num_tasks
        if i < num_tasks
            push!(ranges, (i-1) * (length(tcodes) ÷ num_tasks) + 1 : (i * (length(tcodes) ÷ num_tasks)))
        else
            push!(ranges, (i-1) * (length(tcodes) ÷ num_tasks) + 1 : length(tcodes))
        end
    end

    # launch each of the asynchronous tasks with the segments of data each
    # task operates on as well as a unique file name to write to.
    @sync for (i, range) in enumerate(ranges)
            outfile = string(splitext(out_file_name)[1], "_", i, splitext(out_file_name)[2])
            optimize() = optimize_directed_graph_guesses(guess_geoms[range], label[:], potential_constructor(), out_file_name=outfile)
            # launch the optimization task
            @async optimize()
    end
end