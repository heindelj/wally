include("nwchem_input_generator.jl")

function write_multi_input_file(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, rem_input::String, charge::Int, multiplicity::Int)
    """
    Writes a Q-Chem input file for multiple jobs to be run sequentially.
    rem_input is a file containing the rem block and any other blocks needed
    for the desired calculation.
    """ 
    used_input_name = next_unique_name(infile_name)
    rem_input_string = read(rem_input, String)
    open(used_input_name, "w") do io
        for i in 1:length(geoms)
            geom_string = geometry_to_string(geoms[i], labels[i])
            write(io, "\$molecule\n")
            write(io, string(charge, " ", multiplicity, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string)
            write(io, "\n@@@\n\n")
        end    
    end
end
