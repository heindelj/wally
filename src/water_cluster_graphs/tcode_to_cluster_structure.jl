using Distributed
using GraphIO

include("../molecule_tools/water_tools.jl")
include("../molecule_tools/read_xyz.jl")
include("../molecule_tools/molecular_axes.jl") # centroid
include("../molecule_tools/optimize_xyz.jl")
include("molecular_graph_utils.jl")
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

function form_directed_molecular_graph(t_code::tcode)
    G = SimpleDiGraph(t_code.indices[1])
    for i in 3:2:length(t_code.indices)
        add_edge!(G, t_code.indices[i], t_code.indices[i+1])
    end
    return G
end

function form_directed_molecular_graph(t_codes::AbstractVector{tcode})
    return form_directed_molecular_graph.(t_codes)
end

function get_hbond_partners_from_tcode(tcode_array::tcode)
    """
    Takes a tcode and splits it into tuples which represent the donor and acceptor
    oxygen atoms, respectively.
    Note that the first two numbers tell you how many edges and vertices there are,
    which is not neccesary for connectivity.
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

function dangling_hydrogen_from_centroid(vec_O::AbstractVector, centroid::AbstractVector; free_OH_distance::Float64 = 0.95)
    line = vec_O - centroid
    line = line * (norm(line) + free_OH_distance) / norm(line)
    return centroid + line
end

function construct_dodecahedral_cage(desired_edge_distance::Float64)
    """
    Uses the analytic cartesian coordinates for a dodecahedron to create
    the oxygen coordinates with desired edge lengths.
    See wikipedia for coordinates: https://en.wikipedia.org/wiki/Regular_dodecahedron
    """
    ϕ = (1.0 + sqrt(5.0)) / 2.0
    original_edge_length = sqrt(5.0) - 1.0
    vertices::HybridArray{Tuple{3,StaticArrays.Dynamic()}} = [[1,1,1] [1,1,-1] [1,-1,1] [-1,1,1] [1,-1,-1] [-1,1,-1] [-1,-1,1] [-1,-1,-1] [0,ϕ,1/ϕ] [0,ϕ,-1/ϕ] [0,-ϕ,1/ϕ] [0,-ϕ,-1/ϕ] [1/ϕ,0,ϕ] [1/ϕ,0,-ϕ] [-1/ϕ,0,ϕ] [-1/ϕ,0,-ϕ] [ϕ,1/ϕ,0] [ϕ,-1/ϕ,0] [-ϕ,1/ϕ,0] [-ϕ,-1/ϕ,0]]

    return vertices * desired_edge_distance / original_edge_length
end

function get_g6(coords::AbstractMatrix, nn_distance::Float64)
    adj_matrix = zeros((size(coords, 2), size(coords, 2)))
    for i_col in 1:size(coords, 2)
        for i_nn in 1:size(coords, 2)
            if i_col != i_nn
                if norm(coords[:, i_col] - coords[:, i_nn]) < nn_distance + 0.1
                    adj_matrix[i_col, i_nn] = 1
                end
            end
        end
    end
    g6_string = GraphIO.Graph6._graphToG6String(LightGraphs.SimpleGraph(adj_matrix))
    return g6_string
end

function structure_from_tcode(t_code::tcode; OO_distance::Float64 = 2.65, OH_distance::Float64 = 0.98, free_OH_distance::Float64 = 0.95)
    """
    This is a special case for the (H2O)20 dodecahedral cage.

    NOTE: This only works when the reference structure used to generate the 
    t codes has the same atom ordering as ordering returned by construct_dodecahedral_cage
    which is somewhat arbitrary. Consider yourself warned!
    """
    structure = zeros((3,60))
    O_indices = collect(1:3:60)
    structure[:, O_indices] = construct_dodecahedral_cage(OO_distance)

    return structure_from_tcode(t_code, structure, OH_distance=OH_distance, free_OH_distance=free_OH_distance)
end

function structure_from_tcode!(t_code::tcode, ref_coords::AbstractMatrix, structure::AbstractMatrix; OH_distance::Float64=0.98, free_OH_distance::Float64=0.95)
    # get the oxygen atom positions at the approproate indices
    remaining_indices = collect(1:size(ref_coords)[2])
    used_indices = collect(1:3:size(ref_coords)[2]) # start with the oxygen indices
    @views structure[:, used_indices] = ref_coords[:, used_indices]

    # place the hydrogen-bonded h atoms
    pairs = get_hbond_partners_from_tcode(t_code)
    @inbounds for (i, pair) in enumerate(pairs)
        @views h_vec = structure[:, 3 * pair[2] - 2] - structure[:, 3 * pair[1] - 2]
        t = OH_distance / norm(h_vec)
        if iszero(@view(structure[:, (3 * pair[1] - 2) + 1]))
            @views structure[:, (3 * pair[1] - 2) + 1] = structure[:, 3 * pair[1] - 2] + t * h_vec
            push!(used_indices, (3 * pair[1] - 2) + 1)
        else
            @views structure[:, (3 * pair[1] - 2) + 2] = structure[:, 3 * pair[1] - 2] + t * h_vec
            push!(used_indices, (3 * pair[1] - 2) + 2)
        end
    end
    # get the indices of free OH atoms
    setdiff!(remaining_indices, used_indices)

    # place the free OH atoms (I slightly randomize so we don't get colinear vectors)
    for i_free_OH in remaining_indices
        @views structure[:, i_free_OH] = dangling_hydrogen_from_centroid(structure[:, 3 * ((i_free_OH - 1) ÷ 3) + 1], vec(centroid(ref_coords)), free_OH_distance=free_OH_distance) + rand(3) * 0.0005
    end

    # make all HOH angles 104.5
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
end

function structure_from_tcode(t_code::tcode, ref_coords::AbstractMatrix; OH_distance::Float64 = 0.98, free_OH_distance::Float64 = 0.95)
    """
    Takes a tcode and set of reference coordinates for water molecules in OHH order.
    Returns a complete water cluster structure.
    """
    structure = zero(ref_coords)
    structure_from_tcode!(t_code, ref_coords, structure, OH_distance=OH_distance, free_OH_distance=free_OH_distance)
    return structure
end

function structures_from_tcode(t_codes::AbstractArray{tcode, 1}, ref_coords::AbstractMatrix; OH_distance::Float64 = 0.98, free_OH_distance::Float64 = 0.95)
    """
    Make many structures from the relevant tcodes and reference geometry.
    """
    out_structures = [zero(ref_coords) for _ in 1:length(t_codes)]
    Threads.@threads for i in 1:length(t_codes)
        structure_from_tcode!(t_codes[i], ref_coords, out_structures[i], OH_distance=OH_distance, free_OH_distance=free_OH_distance)
    end
    return out_structures
    #return structure_from_tcode.(t_codes, (ref_coords,), OH_distance=OH_distance, free_OH_distance=free_OH_distance)
end

function structures_from_tcode(t_codes::AbstractArray{tcode, 1}; OO_distance::Float64 = 2.65, OH_distance::Float64 = 0.98, free_OH_distance::Float64 = 0.95)
    """
    Specifically makes many of the generated dodecahedron geometries.
    """
    return structure_from_tcode.(t_codes, OO_distance=OO_distance, OH_distance=OH_distance, free_OH_distance=free_OH_distance)
end

function optimize_directed_graph_guesses(guess_geoms::AbstractArray{Array{Float64, 2}, 1}, labels::AbstractArray{Array{String, 1}}, potential::AbstractPotential; write_every::Union{Int,Nothing}=nothing, out_file_name::AbstractString=string(pwd(), "/optimized_structures.xyz"))
    """
    Optimize the actual guess structures and write the results to specified
    output file.
    Write every will split the calculation into chunks and write results to a different file for each chunk.
    Note that the parallelism is over these chunks, so they should be large enough that the processors have sufficient work.
    """
    if write_every === nothing
        energies, opt_geoms = optimize_structures(guess_geoms, potential, copy_construct_potential=true)
        
        header::String = ""
        for i in 1:length(energies)
            header = string(size(opt_geoms[i], 2), "\n", energies[i])
            write_xyz(out_file_name, [header], labels, [opt_geoms[i]], append=(i!=1))
        end
    else
        ranges = []
        # get the ranges over which we chunk the data
        num_chunks::Int = (length(guess_geoms) ÷ write_every)
        for i in 1:num_chunks
            if i < num_chunks
                push!(ranges, (i-1) * (length(guess_geoms) ÷ num_chunks) + 1 : (i * (length(guess_geoms) ÷ num_chunks)))
            else
                push!(ranges, (i-1) * (length(guess_geoms) ÷ num_chunks) + 1 : length(guess_geoms))
            end
        end
        
        header = ""
        for i in 1:num_chunks
            energies, opt_geoms = optimize_structures(guess_geoms[ranges[i]], potential, copy_construct_potential=true)
            
            outfile = string(splitext(out_file_name)[1], "_", "0"^(length(digits(num_chunks)) - length(digits(i))), i, splitext(out_file_name)[2])
            for j in 1:length(energies)
                header = string(size(opt_geoms[j], 2), "\n", energies[j])
                write_xyz(outfile, [header], labels, [opt_geoms[j]], append=(j!=1))
            end
        end
    end
end

function optimize_directed_graph_guesses(tcode_file::AbstractString, ref_structure_file::AbstractString, potential::AbstractPotential; out_file_name::AbstractString=string(pwd(), "/optimized_structures.xyz"), use_reference_for_guess=true)
    """
    convenience function for calling the overload of this function but takes to read in the relevant data.
    """

    println("Reading in the tcodes and reference structure...")
    tcodes = load_tcode_file(tcode_file)
    header, label, ref_geom = read_xyz(ref_structure_file)

    println("Forming the guess structures...")
    if use_reference_for_guess
        guess_geoms = structures_from_tcode(tcodes, ref_geom[1])
    else
        guess_geoms = structures_from_tcode(tcodes)
    end

    println("Optimizing the guess structures...")
    optimize_directed_graph_guesses(guess_geoms, 
                                    label, 
                                    potential, 
                                    out_file_name=out_file_name)

end
