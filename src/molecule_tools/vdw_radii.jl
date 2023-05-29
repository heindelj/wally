
function vdw_radius(atom_label::String)
    vdw_radii = Dict(
        "H"  => 1.10,
        "He" => 1.40,
        "Li" => 1.81,
        "Be" => 1.53,
        "B"  => 1.92,
        "C"  => 1.70,
        "N"  => 1.55,
        "O"  => 1.52,
        "F"  => 1.47,
        "Ne" => 1.54,
        "Na" => 2.27,
        "Mg" => 1.73,
        "Al" => 1.84,
        "Si" => 2.10,
        "P"  => 1.80,
        "S"  => 1.80,
        "Cl" => 1.75,
        "Ar" => 1.88,
        "K"  => 2.75,
        "Ca" => 2.31,
        "Ga" => 1.87,
        "Ge" => 2.11,
        "As" => 1.85,
        "Se" => 1.90,
        "Br" => 1.83,
        "Kr" => 2.02,
        "Rb" => 3.03,
        "Sr" => 2.49,
        "In" => 1.93,
        "Sn" => 2.17,
        "Sb" => 2.06,
        "Te" => 2.06,
        "I"  => 1.98,
        "Xe" => 2.16,
        "Cs" => 3.43,
        "Ba" => 2.68,
        "Tl" => 1.96,
        "Pb" => 2.02,
        "Bi" => 2.07,
        "Po" => 1.97,
        "At" => 2.02,
        "Rn" => 2.20,
        "Fr" => 3.48,
        "Ra" => 2.83,
        )
    return vdw_radii[titlecase(atom_label)]
end

function ionic_radius(atom_label::String)
    ionic_radii = Dict(
        "F"  => 1.33,
        "Cl" => 1.81,
        "Br" => 1.96,
        "I"  => 2.60, # made up
        "Li" => 0.76,
        "Na" => 1.02,
        "K"  => 1.38,
        "Rb" => 1.72, # made up
        "Cs" => 2.10, # made up
        "Be" => 0.45,
        "Mg" => 0.82  # made up
    )
    if haskey(ionic_radii, atom_label)
        return ionic_radii[titlecase(atom_label)]
    end
    return 0.0
end