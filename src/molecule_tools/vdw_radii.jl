
function vdw_radius(atom_label::String)
    vdw_radii = Dict(
                 "H"   => 1.10,
                 "He"  => 1.40,
                 "Li"  => 0.76,
                 "Be" => 0.59, 
                 "B"   => 1.92,
                 "C"   => 1.70,
                 "N"   => 1.55,
                 "O"  => 1.52,
                 "F"   => 1.47,
                 "Ne"  => 1.54,
                 "Na"  => 1.02,
                 "Mg" => 0.86, 
                 "Al"  => 1.84,
                 "Si"  => 2.10,
                 "P"   => 1.80,
                 "S"  => 1.80, 
                 "Cl"  => 1.81,
                 "Ar"  => 1.88,
                 "K"   => 1.38,
                 "Ca" => 1.14, 
                 "Sc"  => 2.11,
                 "Ti"  => 2.00,
                 "V"   => 2.00,
                 "Cr" => 2.00, 
                 "Mn"  => 2.00,
                 "Fe"  => 2.00,
                 "Co"  => 2.00,
                 "Ni" => 1.63, 
                 "Cu"  => 1.40,
                 "Zn"  => 1.39,
                 "Ga"  => 1.87,
                 "Ge" => 2.11, 
                 "As"  => 1.85,
                 "Se"  => 1.90,
                 "Br"  => 1.85,
                 "Kr" => 2.02, 
                 "Rb"  => 3.03,
                 "Sr"  => 2.49,
                 "Y"   => 2.00,
                 "Zr" => 2.00, 
                 "Nb"  => 2.00,
                 "Mo"  => 2.00,
                 "Tc"  => 2.00,
                 "Ru" => 2.00, 
                 "Rh"  => 2.00,
                 "Pd"  => 1.63,
                 "Ag"  => 1.72,
                 "Cd" => 1.58, 
                 "In"  => 1.93,
                 "Sn"  => 2.17,
                 "Sb"  => 2.06,
                 "Te" => 2.06, 
                 "I"   => 1.98,
                 "Xe"  => 2.16,
                 "Cs"  => 1.67,
                 "Ba" => 1.49, 
                 "La"  => 2.00,
                 "Ce"  => 2.00,
                 "Pr"  => 2.00,
                 "Nd" => 2.00, 
                 "Pm"  => 2.00,
                 "Sm"  => 2.00,
                 "Eu"  => 2.00,
                 "Gd" => 2.00, 
                 "Tb"  => 2.00,
                 "Dy"  => 2.00,
                 "Ho"  => 2.00,
                 "Er" => 2.00, 
                 "Tm"  => 2.00,
                 "Yb"  => 2.00,
                 "Lu"  => 2.00,
                 "Hf" => 2.00, 
                 "Ta"  => 2.00,
                 "W"   => 2.00,
                 "Re"  => 2.00,
                 "Os" => 2.00, 
                 "Ir"  => 2.00,
                 "Pt"  => 1.75,
                 "Au"  => 1.66,
                 "Hg" => 1.55, 
                 "Tl"  => 1.96,
                 "Pb"  => 2.02,
                 "Bi"  => 2.07,
                 "Po" => 1.97, 
                 "At"  => 2.02,
                 "Rn"  => 2.20,
                 "Fr"  => 3.48,
                 "Ra" => 2.83, 
                 "Ac"  => 2.00,
                 "Th"  => 2.00,
                 "Pa"  => 2.00,
                 "U"  => 1.86, 
                 "Np"  => 2.00,
                 "Pu"  => 2.00,
                 "Am"  => 2.00,
                 "Cm" => 2.00, 
                 "Bk"  => 2.00,
                 "Cf"  => 2.00,
                 "Es"  => 2.00,
                 "Fm" => 2.00, 
                 "Md"  => 2.00,
                 "No"  => 2.00,
                 "Lr"  => 2.00,
                 "Rf" => 2.00, 
                 "Db"  => 2.00,
                 "Sg"  => 2.00,
                 "Bh"  => 2.00,
                 "Hs" => 2.00, 
                 "Mt"  => 2.00,
                 "Ds"  => 2.00,
                 "Rg"  => 2.00,
                 "Cn" => 2.00, 
                 "Uut" => 2.00,
                 "Fl"  => 2.00,
                 "Uup" => 2.00,
                 "Lv" => 2.00,
                 "Uus" => 2.00, 
                 "Uuo" => 2.00
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