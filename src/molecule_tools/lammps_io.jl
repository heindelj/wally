include("atomic_masses.jl")

function write_reaxff_cgem_input(ofile::String, coords::AbstractMatrix, labels::AbstractVector{String}, index_and_charge::Union{Dict{Int, Int}, Nothing}=nothing)
    """
    Writes a reaxff/CGEM input for the modified version of Lammps.
    index_and_charge is a dictionary specifying which index should have an additional
    shell placed at it. i.e. (2,-1) would put an extra electronic shell at atom 2.
    """
    labels_unique = unique(labels)
    masses = atomic_masses(labels_unique)
    
    label_to_index = Dict{String, Int}()
    for i in 1:length(labels_unique)
     label_to_index[labels_unique[i]] = i
    end

    open(ofile, "w") do io
     write(io, "# Comment line\n\n")
     write(io, string(2 * length(labels) + ((index_and_charge === nothing) ? 0 : length(keys(index_and_charge))), " atoms\n"))
     write(io, string(length(labels_unique) + 1, " atom types\n")) #+1 for the electron shell
     write(io, "\n-250 250 xlo xhi\n")
     write(io, "-250 250 ylo yhi\n")
     write(io, "-250 250 zlo zhi\n")
     write(io, "\nMasses\n\n")
     for i in 1:length(masses)
             write(io, string(i, " ", masses[i], "\n"))
     end
     write(io, string(length(masses)+1, " 0.01\n")) # electron shell
     write(io, "\nAtoms\n\n")

     # write actual atom and shell for each atom
     count = 1
     for (i, vec) in enumerate(eachcol(coords))
             write(io, string(count, "   ", label_to_index[labels[i]], "  1.0 ", vec[1], " ", vec[2], " ", vec[3], "\n"))
             count += 1
             write(io, string(count, "   3 -1.0 ", vec[1]+0.1, " ", vec[2]+0.1, " ", vec[3]+0.1, "\n"))
             count += 1
             if index_and_charge !== nothing && i in keys(index_and_charge)
                     write(io, string(count, "   3 ", Float64(get(index_and_charge, i, -1)), " ", vec[1]-0.1, " ", vec[2]-0.1, " ", vec[3]-0.1, "\n"))
                     count += 1
             end
     end
    end
end
