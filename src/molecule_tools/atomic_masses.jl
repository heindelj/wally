include("units.jl")

function label_to_mass(atom_label::Symbol)
    masses = Dict(
        :O => 15.99491561957,
        :H => 1.00782503223 ,
        :D => 2.0141017778  ,
        :C => 11.9999999958 ,
        :N => 14.003074
        )
    return masses[atom_label]
end

function atomic_masses(atom_labels::Vector{String})
    masses = zeros(length(atom_labels))
    for i in eachindex(atom_labels)
        masses[i] = label_to_mass(Symbol(atom_labels[i]))
    end
    return masses
end