include("qchem_input_generator.jl")
include("read_xyz.jl")
include("molecular_cluster.jl")
using ProgressMeter

function write_fchk_input_files(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, rem_input::String)
    write_fchk_input_files(infile_name, geoms, labels, rem_input, zeros(Int, length(geoms)), ones(Int, length(geoms)))
end

function write_fchk_input_files(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, rem_input::String, fragment_charges::Vector{Int}, fragment_multiplicites::Vector{Int})
    if !isdir("fchk_files")
        mkdir("fchk_files")
    end
    write_separate_input_files_for_fragments(infile_name, geoms, labels, rem_input, fragment_charges, fragment_multiplicites, "fchk_files")
end

function write_fchk_input_files_multiple_fragmented_geoms(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, rem_input::String, fragment_indices::Vector{Vector{Int}})
    write_fchk_input_files_multiple_fragmented_geoms(infile_name, geoms, labels, rem_input, fragment_indices, zeros(Int, length(fragment_indices)), ones(Int, length(fragment_indices)))
end

function write_fchk_input_files_multiple_fragmented_geoms(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, rem_input::String, fragment_indices::Vector{Vector{Int}}, fragment_charges::Vector{Int}, fragment_multiplicites::Vector{Int})
    infile_prefix = splitext(infile_name)[1]
    for i in eachindex(geoms)
        fragmented_geom = [geoms[i][:, indices] for indices in fragment_indices]
        fragment_labels = [labels[i][indices] for indices in fragment_indices]
        write_fchk_input_files(string(infile_prefix, "_geom_", i, ".in"), fragmented_geom, fragment_labels, rem_input, fragment_charges, fragment_multiplicites)
    end
end

function write_and_run_gdma_input_file(fchk_file::String, output_dir::Union{Nothing, String}=nothing)
    file_prefix = splitext(fchk_file)[1]
    relative_path_to_fchk = file_prefix
    if output_dir !== nothing
        output_dir = strip(output_dir, '/')
        mkpath(output_dir)
        relative_path_to_fchk = string("../", file_prefix)
    else
        output_dir = "."
    end
    punch_file = splitext(basename(file_prefix))[1]
    cd(output_dir)
    infile = string(basename(file_prefix), ".gdma.in")
    open(infile, "w") do io
        write(io, "Title $file_prefix.fchk\n")
        write(io, "File $relative_path_to_fchk.fchk\n")
        write(io, "Angstrom\n")
        write(io, "Multipoles\n")
        write(io, "  switch 4\n")
        write(io, "  Limit 4\n")
        write(io, "  Limit 1 H\n")
        write(io, "  Radius H 0.325\n")
        write(io, "  Punch $punch_file.punch\n")
        write(io, "Start\n")
        write(io, "Finish\n")
    end
    gdma_output = read(pipeline(`gdma`, stdin=infile), String)
    outfile = string(splitext(infile)[1], ".out")
    open(outfile, "w") do io
        write(io, gdma_output)
    end
    punch_file_lines = readlines(string(splitext(splitext(outfile)[1])[1], ".punch"))
    for i in eachindex(punch_file_lines)
        if occursin("Units", punch_file_lines[i])
            punch_file_lines[i] = string("! ", punch_file_lines[i])
            break
        end
    end

    open(string(splitext(splitext(outfile)[1])[1], ".punch"), "w") do io
        for i in eachindex(punch_file_lines)
            write(io, string(punch_file_lines[i], "\n"))
        end
    end

    cd("..")
end

function write_and_run_many_gdma_input_files(fchk_dir::String)
    output_dir = "gdma_files"
    mkpath(output_dir)
    fchk_dir = strip(fchk_dir, '/')
    all_files = readdir(fchk_dir)

    all_fchk_files = String[]
    for i in eachindex(all_files)
        if occursin(".fchk", all_files[i])
            push!(all_fchk_files, string(fchk_dir, "/", all_files[i]))
        end
    end

    @showprogress pmap((x) -> write_and_run_gdma_input_file(x, output_dir), all_fchk_files)
end

function write_and_run_orient_input_file(punch_files::Vector{String}, output_dir::Union{Nothing,String}=nothing)
    file_prefix = splitext(punch_files[1])[1]
    if output_dir !== nothing
        output_dir = strip(output_dir, '/')
        mkpath(output_dir)
    else
        output_dir = "."
    end
    cd(output_dir)
    m = match(r"[^_]+$", basename(file_prefix))
    infile = string(basename(file_prefix)[1:m.offset-2], ".orient.in")
    open(infile, "w") do io
        write(io, "Types\n")
        write(io, "  O Z 8\n")
        write(io, "  H Z 1\n")
        write(io, "End\n")
        write(io, "Units ANGSTROM\n")
        for i in eachindex(punch_files)
            punch_file = string("../", punch_files[i])
            write(io, string("Molecule Mol_", i, "\n"))
            write(io, "  #include $punch_file\n")
            write(io, "End\n")
        end
        write(io, "Units KJ/MOL\n")
        write(io, "Energy\n")
        write(io, "Finish\n")
    end

    orient_output = read(pipeline(`orient`, stdin=infile), String)
    outfile = string(splitext(infile)[1], ".out")
    open(outfile, "w") do io
        write(io, orient_output)
    end

    cd("..")

    lines = split(orient_output, '\n')
    for line in lines
        if occursin("Electrostatic energy", line)
            return tryparse(Float64, split(line)[end-1])
        end
    end
    return nothing
end

function write_and_run_many_orient_input_files(gdma_dir::String)
    output_dir = "orient_files"
    mkpath(output_dir)
    gdma_dir = strip(gdma_dir, '/')
    all_files = readdir(gdma_dir)

    
    all_gdma_files = String[]
    for i in eachindex(all_files)
        if occursin(".punch", all_files[i])
            push!(all_gdma_files, string(gdma_dir, "/", all_files[i]))
        end
    end

    geom_nums = Int[]
    for i in eachindex(all_gdma_files)
        m = match(r"_[0-9]+_", all_gdma_files[i])
        geom_num = tryparse(Int, string(match(r"[0-9]+", m.match).match))
        push!(geom_nums, geom_num)
    end
    grouped_gdma_files = [String[] for _ in 1:length(Set(geom_nums))]
    for i in eachindex(geom_nums)
        push!(grouped_gdma_files[geom_nums[i]], all_gdma_files[i])
    end

    @showprogress pmap((x) -> write_and_run_orient_input_file(x, output_dir), grouped_gdma_files)
end
