include("units.jl")

function label_to_mass(atom_label::Symbol)
    masses = Dict(
        :H  => 1.00782503223,
        :D  => 2.0141017778,
        :C  => 11.9999999958,
        :N  => 14.003074,
        :O  => 15.99491561957,
        :Li => 6.96,
        :Na => 22.98976928,
        :K  => 39.0983,
        :Rb => 85.4678,
        :Cs => 132.90545196,
        :Be => 9.0121831,
        :Mg => 24.3055,
        :Ca => 40.078,
        :F  => 18.998403163,
        :Cl => 35.450,
        :Br => 79.904,
        :I  => 126.90447,   
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
