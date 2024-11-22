using Printf
include("read_xyz.jl")

function geometry_to_string(geom::AbstractMatrix{Float64}, atom_labels::AbstractVector{String})
    geom_string = ""
    for (i, vec) in enumerate(eachcol(geom))
        geom_string = string(geom_string, atom_labels[i], " ", join(vec, " "), "\n")
    end
    return geom_string
end

"""
Takes an xyz file, possibly containing multiple geometries, and
converts this to the nrg format which splits things into fragments.
We will assume that all oxygen atoms denote the beginning of a
water molecule. We will also look for ions based on their labels.
A more general implementation will require passing the connectivity
information explicitly which, because I don't need that right now,
I am not writing.
"""
function convert_xyz_to_nrg_files(xyz_file::String)
    _, labels, geoms = read_xyz(xyz_file)
    num_geoms = length(geoms)

    molecule_labels = Dict(
        "O" => "h2o",
        "F"  => "f-",
        "Cl" => "cl-",
        "Br" => "br-",
        "I"  => "i-",
        "Li" => "li+",
        "Na" => "na+",
        "K"  => "k+",
        "Rb" => "rb+",
        "Cs" => "cs+"
    )

    nrg_file_prefix = splitext(xyz_file)[1]

    for i_geom in eachindex(geoms)
        i_geom_string = string(i_geom)
        #if i_geom < 10
        #    i_geom_string = string("0", i_geom)
        #end
        nrg_file = string(nrg_file_prefix, "_", i_geom_string, ".nrg")
        if num_geoms == 1
            nrg_file = string(nrg_file_prefix, ".nrg")
        end

        file_as_array_of_strings = ["SYSTEM\n"]
        for i in eachindex(labels[i_geom])
            if labels[i_geom][i] == "H"
                continue
            end
            mol_label = molecule_labels[titlecase(labels[i_geom][i])]
            push!(file_as_array_of_strings, string("MOLECULE\nMONOMER ", mol_label, "\n"))
            if mol_label == "h2o"
                push!(file_as_array_of_strings, geometry_to_string(geoms[i_geom][:, i:i+2], labels[i_geom][i:i+2]))
            else
                push!(file_as_array_of_strings, geometry_to_string(reshape(geoms[i_geom][:, i], 3, 1), [labels[i_geom][i]]))
            end
            push!(file_as_array_of_strings, "ENDMON\nENDMOL\n")
        end
        push!(file_as_array_of_strings, "ENDSYS\n")
        open(nrg_file, "w") do io
            full_string = join(file_as_array_of_strings)
            write(io, full_string)
        end
    end
end