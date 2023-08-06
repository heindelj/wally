using StaticArrays, NearestNeighbors, ProgressBars, ProgressMeter, DelimitedFiles
using StatsBase: countmap
include("covalent_radii.jl")
include("molecular_axes.jl")
include("read_xyz.jl")
include("water_tools.jl")
include("qchem_input_generator.jl")
include("molecular_cluster.jl")

function generate_cgem_optimization_script()
    return "from cgem.model import CGem
from cgem.utils import load_xyz
from cgem.utils import compute_stats
from cgem.attypes import classify_carbon
from cgem.parameters import get_drug_parameters
from cgem.molecules import add_shell_in_place
import numpy as np
import sys, os

def write_charges(coords_cores, coords_shells, ofile=None):
    output = \"\"
    for i in range(len(coords_cores)):
        output += \"\".join(map(\'{:.8f} \'.format, coords_cores[i]))
        output += \'1.0 \\n\'
    for i in range(len(coords_shells)):
        output += \"\".join(map(\'{:.8f} \'.format, coords_shells[i]))
        output += \'-1.0 \\n\'
    if ofile is None:
        print(output)
    else:
        with open(ofile, \'w\') as f:
            f.write(output)

def nums_to_labels(nums):
    num_to_label_dict = {
        1: \"H\",
        8: \"O\"
    }
    return [num_to_label_dict[num] for num in nums]

if __name__ == \'__main__\':
    try:
        cluster_file, env_file, num_hydroxides = sys.argv[1], sys.argv[2], sys.argv[3]
    except:
        print(\"Expected cluster xyz file followed by environment xyz file followed by index for writing output charges.\")
        sys.exit(1)

    num_hydroxides = int(num_hydroxides)
    cluster_nums, cluster_coords = load_xyz(cluster_file)
    env_nums, env_coords = load_xyz(env_file)
    all_coords = np.vstack((cluster_coords, env_coords))
    all_nums = list(cluster_nums) + list(env_nums)
    drug_params = get_drug_parameters()
    for i in range(2*(num_hydroxides-2), -1, -2):
        env_coords = add_shell_in_place(env_coords, i)
    coords_s = np.vstack((cluster_coords, env_coords))
    cgem = CGem.from_molecule(all_nums, all_coords, coords_s=coords_s, opt_shells=True, **drug_params)
    write_charges(cgem.coords_c[len(cluster_coords):,:], cgem.coords_s[len(cluster_coords):,:], os.path.splitext(cluster_file)[0] + \"_shell_positions_radical.txt\")
    coords_s = add_shell_in_place(cgem.coords_s, 0)
    cgem = CGem.from_molecule(all_nums, all_coords, coords_s=coords_s, opt_shells=True, **drug_params)
    write_charges(cgem.coords_c[len(cluster_coords):,:], cgem.coords_s[(len(cluster_coords)+1):,:], os.path.splitext(cluster_file)[0] + \"_shell_positions_anion.txt\")"
end

function set_up_shell_position_optimization_for_sampled_clusters_and_environments(
    cluster_file::String,
    environment_file::String
)

    _, cluster_labels, cluster_geoms = read_xyz(cluster_file)
    _, env_labels, env_geoms = read_xyz(environment_file)
    @assert length(cluster_geoms) == length(env_geoms)
    
    mkpath("sampled_geoms_and_optimized_shells")
    opt_script = generate_cgem_optimization_script()
    open("sampled_geoms_and_optimized_shells/optimize_shells.py", "w") do io
        write(io, opt_script)
    end
    for i in eachindex(cluster_geoms)
        write_xyz(string("sampled_geoms_and_optimized_shells/cluster_sample_", i, ".xyz"), cluster_labels[i], cluster_geoms[i])
        write_xyz(string("sampled_geoms_and_optimized_shells/env_sample_", i, ".xyz"), env_labels[i], env_geoms[i])
    end
    #for i in eachindex(cluster_geoms)
    #    smaller_cluster_labels, smaller_cluster_geom, extra_env_labels, extra_env_geom = find_n_nearest_neighbors(build_cluster(cluster_geoms[i], cluster_labels[i]), 1, 20)
    #    write_xyz(string("sampled_geoms_and_optimized_shells/cluster_sample_mp2_", i, ".xyz"), smaller_cluster_labels, smaller_cluster_geom)
    #    write_xyz(string("sampled_geoms_and_optimized_shells/env_sample_mp2_", i, ".xyz"), vcat(env_labels[i], extra_env_labels), hcat(env_geoms[i], extra_env_geom))
    #end
end

"""
This function assumes that you have generated cluster_samples and the
charges which are used as the environment. All geometries and charges are
read from ./sampled_geoms_and_optimized_shells/
"""
function write_input_files_for_vie_qchem(
    qchem_infile_prefix::String,
    num_samples::Int
)
    atom_charges = Dict(
        "O"   => -2,
        "Cl"  => -1,
        "H"   =>  1,
        "Na"  =>  1,
    )

    fragment_basis_sets = Dict(
        ["O", "H"] => "aug-cc-pvtz",
        ["O", "H", "H", "H"] => "aug-cc-pvtz",
        ["Cl"] => "aug-cc-pvtz",
        ["Na"] => "aug-cc-pvtz"
    )
    
    rem_input_string_gas_phase = "\$rem
jobtype                 sp
method                  wB97M-V
unrestricted            1
basis                   mixed
xc_grid        2
scf_max_cycles          500
scf_convergence         6
thresh                  14
s2thresh 14
symmetry                0
sym_ignore              1
mem_total 256000
mem_static 16000
\$end"

    rem_input_string_with_env = "\$rem
jobtype                 sp
method                  wB97M-V
unrestricted            1
basis                   mixed
xc_grid        2
scf_max_cycles          500
scf_convergence         6
SCF_GUESS               read
thresh                  14
s2thresh 14
symmetry                0
sym_ignore              1
mem_total 256000
mem_static 16000
\$end"

    mkpath("qchem_input_files")
    @showprogress for i_sample in 1:num_samples
        if (
            !ispath(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, "_shell_positions_anion.txt"))
        )
            @warn "Couldn't open charges file for sample $i_sample. Moving on to the next sample."
            continue
        end
        _, cluster_labels, cluster_geom = read_xyz(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, ".xyz"))

        cluster_charge = sum([atom_charges[label] for label in cluster_labels[1]])
        
        cluster = build_cluster(cluster_geom[1], cluster_labels[1])
        
        # stores label, atom number, and basis set
        all_basis_sets = Tuple{String, Int, String}[]
        for i_frag in eachindex(cluster.indices)
            if haskey(fragment_basis_sets, cluster.labels[cluster.indices[i_frag]])
                for index in cluster.indices[i_frag]
                    push!(all_basis_sets, (cluster.labels[index], index, fragment_basis_sets[cluster.labels[cluster.indices[i_frag]]]))
                end
            else
                for index in cluster.indices[i_frag]
                    push!(all_basis_sets, (cluster.labels[index], index, "aug-cc-pvdz"))
                end
            end
        end

        basis_string = ""
        for i_atom in eachindex(all_basis_sets)
            basis_string = string(basis_string, all_basis_sets[i_atom][1], " ", all_basis_sets[i_atom][2], "\n", all_basis_sets[i_atom][3], "\n****\n")
        end

        geom_string = geometry_to_string(cluster_geom[1], cluster_labels[1])
        # write anion file
        open(string("qchem_input_files/", qchem_infile_prefix, "_anion_sample_", i_sample, ".in"), "w") do io
            write(io, "\$molecule\n")
            write(io, string(cluster_charge, " ", 1, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_gas_phase)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n@@@\n\n"))
            write(io, "\$molecule\n")
            write(io, string(cluster_charge, " ", 1, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_with_env)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n"))
            write(io, "\n\$external_charges\n")
            writedlm(io, readdlm(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, "_shell_positions_anion.txt")))
            write(io, "\$end\n\n")
        end

        # write radical file
        open(string("qchem_input_files/", qchem_infile_prefix, "_radical_sample_", i_sample, ".in"), "w") do io
            write(io, "\$molecule\n")
            write(io, string(cluster_charge+1, " ", 2, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_gas_phase)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n@@@\n\n"))
            write(io, "\$molecule\n")
            write(io, string(cluster_charge+1, " ", 2, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_with_env)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n"))
            write(io, "\n\$external_charges\n")
            writedlm(io, readdlm(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, "_shell_positions_anion.txt")))
            write(io, "\$end\n\n")
        end
    end
end

function write_input_files_for_water_average_energy(
    infile_prefix::String,
    sampled_geometries_files::String,
    indices_to_exclude::Vector{Int}=Int[]
)

    atom_charges = Dict(
        "O"   => -2,
        "Cl"  => -1,
        "H"   =>  1,
        "Na"  =>  1,
    )

    rem_input_string_gas_phase = "\$rem
jobtype                 sp
method                  wB97M-V
unrestricted            1
basis                   aug-cc-pvdz
xc_grid        2
scf_max_cycles          500
scf_convergence         6
thresh                  14
s2thresh 14
symmetry                0
sym_ignore              1
mem_total 256000
mem_static 16000
SOLVENT_METHOD cosmo
\$end

\$pcm
   Theory          cosmo
\$end\n"
    header, labels, geoms = read_xyz(sampled_geometries_files)
    mkpath("qchem_input_files_enthalpy")
    for i in eachindex(geoms)
        cluster_labels = labels[i][setdiff(1:length(labels[i]), indices_to_exclude)]
        cluster_geom   = geoms[i][:, setdiff(1:size(geoms[i], 2), indices_to_exclude)]
        cluster_charge = sum([atom_charges[label] for label in cluster_labels])
        write_input_file(string("qchem_input_files_enthalpy/", infile_prefix, "_sample_", i, ".in"), cluster_geom, cluster_labels, rem_input_string_gas_phase, cluster_charge, 1)
    end
end

"""
Takes a set of cluster geometries and surrounding environment of charges
and generates q-chem input files for the clusters without a solute of
interest. In this case, we just take indices to remove from the cluster.
In the future, we could generalize this to remove specific molecules, but
we would need some kind of heuristic for when two of the same solute are
in a cluster, so for now we just go with indices.
"""
function write_input_files_for_solvation_enthalpy_qchem(
    qchem_infile_prefix::String,
    num_samples::Int,
    indices_to_remove_from_cluster::Vector{Int}
)
    atom_charges = Dict(
        "O"   => -2,
        "Cl"  => -1,
        "H"   =>  1,
        "Na"  =>  1,
    )

    fragment_basis_sets = Dict(
        ["O", "H"] => "aug-cc-pvtz",
        ["O", "H", "H", "H"] => "aug-cc-pvtz",
        ["Cl"] => "aug-cc-pvtz",
        ["Na"] => "aug-cc-pvtz"
    )
    
    rem_input_string_gas_phase = "\$rem
jobtype                 sp
method                  wB97M-V
unrestricted            1
basis                   mixed
xc_grid        2
scf_max_cycles          500
scf_convergence         6
thresh                  14
s2thresh 14
symmetry                0
sym_ignore              1
mem_total 256000
mem_static 16000
\$end"

    rem_input_string_with_env = "\$rem
jobtype                 sp
method                  wB97M-V
unrestricted            1
basis                   mixed
xc_grid        2
scf_max_cycles          500
scf_convergence         6
SCF_GUESS               read
thresh                  14
s2thresh 14
symmetry                0
sym_ignore              1
mem_total 256000
mem_static 16000
\$end"

    mkpath("qchem_input_files_enthalpy")
    @showprogress for i_sample in 1:num_samples
        if (
            !ispath(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, "_shell_positions_anion.txt"))
        )
            @warn "Couldn't open charges file for sample $i_sample. Moving on to the next sample."
            continue
        end
        _, cluster_labels, cluster_geom = read_xyz(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, ".xyz"))

        cluster_labels[1] = cluster_labels[1][setdiff(1:length(cluster_labels[1]), indices_to_remove_from_cluster)]
        cluster_geom[1]   = cluster_geom[1][:, setdiff(1:size(cluster_geom[1], 2), indices_to_remove_from_cluster)]

        cluster_charge = sum([atom_charges[label] for label in cluster_labels[1]])
        
        cluster = build_cluster(cluster_geom[1], cluster_labels[1])
        
        # stores label, atom number, and basis set
        all_basis_sets = Tuple{String, Int, String}[]
        for i_frag in eachindex(cluster.indices)
            if haskey(fragment_basis_sets, cluster.labels[cluster.indices[i_frag]])
                for index in cluster.indices[i_frag]
                    push!(all_basis_sets, (cluster.labels[index], index, fragment_basis_sets[cluster.labels[cluster.indices[i_frag]]]))
                end
            else
                for index in cluster.indices[i_frag]
                    push!(all_basis_sets, (cluster.labels[index], index, "aug-cc-pvdz"))
                end
            end
        end

        basis_string = ""
        for i_atom in eachindex(all_basis_sets)
            basis_string = string(basis_string, all_basis_sets[i_atom][1], " ", all_basis_sets[i_atom][2], "\n", all_basis_sets[i_atom][3], "\n****\n")
        end

        geom_string = geometry_to_string(cluster_geom[1], cluster_labels[1])
        # write anion file
        open(string("qchem_input_files_enthalpy/", qchem_infile_prefix, "_no_solute_", i_sample, ".in"), "w") do io
            write(io, "\$molecule\n")
            write(io, string(cluster_charge, " ", 1, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_gas_phase)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n@@@\n\n"))
            write(io, "\$molecule\n")
            write(io, string(cluster_charge, " ", 1, "\n"))
            write(io, geom_string)
            write(io, "\$end\n\n")
            write(io, rem_input_string_with_env)
            write(io, string("\n\n\$basis\n", basis_string, "\$end\n\n"))
            write(io, "\n\$external_charges\n")
            writedlm(io, readdlm(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, "_shell_positions_anion.txt")))
            write(io, "\$end\n\n")
        end
    end
end

function write_dimer_inputs_for_non_electrostatic_enthalpy_contribution(
    qchem_infile_name::String,
    cluster_file::String,
    env_file::String,
    center_index::Int=1,
    dimer_sampling_radius::Float64=12.0
)

    atom_charges = Dict(
        "O"   => -2,
        "H"   =>  1,
    )

    _, cluster_labels, cluster_geoms = read_xyz(cluster_file)
    _, env_labels, env_geoms = read_xyz(env_file)

    cluster_labels = cluster_labels[1]
    cluster_geom = cluster_geoms[1]
    env_labels = env_labels[1]
    env_geom = env_geoms[1]

    solvating_cluster = build_cluster(cluster_geom, cluster_labels)
    total_cluster     = build_cluster(hcat(cluster_geom, env_geom), vcat(cluster_labels, env_labels))

    neighbor_indices = find_neighbors_within_range(total_cluster, center_index, dimer_sampling_radius)
    
    # exclude indices in the explicit solvent
    dimer_indices = setdiff(neighbor_indices, 1:length(solvating_cluster.indices))

    rem_string = "\$rem
JOBTYPE eda
SCFMI_FREEZE_SS 2
FRZ_ORTHO_DECOMP -1
EDA_CLS_DISP 1
method wB97M-V
BASIS aug-cc-pvtz
XC_GRID 000099000590
NL_GRID 1
UNRESTRICTED FALSE
MAX_SCF_CYCLES 200
SYMMETRY FALSE
SYM_IGNORE TRUE
mem_total  16000
BASIS_LIN_DEP_THRESH 14
THRESH 14
SCF_CONVERGENCE 8
SCF_PRINT_FRGM TRUE
\$end"

    center_geom = total_cluster.geom[:, total_cluster.indices[center_index]]
    center_labels = total_cluster.labels[total_cluster.indices[center_index]]
    center_charge = sum([atom_charges[label] for label in cluster_labels])
    center_geom_string = geometry_to_string(center_geom, center_labels)
    open(qchem_infile_name, "w") do io
        for i in eachindex(dimer_indices)
            neighbor_geom = total_cluster.geom[:, total_cluster.indices[dimer_indices[i]]]
            neighbor_labels = total_cluster.labels[total_cluster.indices[dimer_indices[i]]]
            neighbor_charge = sum([atom_charges[label] for label in neighbor_labels])
            neighbor_geom_string = geometry_to_string(neighbor_geom, neighbor_labels)

            write(io, "\$molecule\n")
            write(io, string(neighbor_charge + center_charge, " 1\n--\n"))
            write(io, string(center_charge, " 1\n"))
            write(io, center_geom_string)
            write(io, string("--\n", neighbor_charge, " 1\n"))
            write(io, neighbor_geom_string)
            write(io, "\$end\n\n")
            write(io, rem_string)

            if i < length(dimer_indices)
                write(io, "\n\n@@@\n\n")
            end
        end
    end
end

function write_input_files_for_vie_nwchem(
    nwchem_infile_prefix::String,
    num_samples::Int
)
    atom_charges = Dict(
        "O"   => -2,
        "Cl"  => -1,
        "H"   =>  1,
        "Na"  =>  1,
    )

    fragment_basis_sets = Dict(
        ["O", "H"] => "aug-cc-pvtz",
        ["O", "H", "H", "H"] => "aug-cc-pvtz",
        ["Cl"] => "aug-cc-pvtz",
        ["Na"] => "aug-cc-pvtz"
    )
    
    header_string = "echo
start
title \"W20_OH- MP2/AVTZ vertical ionization energy at REAXFF/CGEM Thermalized configurations with point charges from CGeM environment\"\n\nset bq:max_nbq 70000\n"

    method_section_anion = "basis spherical
 O1 library O aug-cc-pvtz
 H1 library H aug-cc-pvtz
 O library aug-cc-pvdz
 H library aug-cc-pvdz
end

set lindep:n_dep 0

scf
vectors atomic output anion.movecs
 thresh 1d-6
 uhf
 singlet
 maxiter 350
 tol2e 1d-15
end

mp2
freeze atomic
end

task mp2 energy

set bq anion

scf
vectors input anion.movecs
 thresh 1d-6
 uhf
 singlet
 maxiter 350
 tol2e 1d-15
end

mp2
freeze atomic
end

task mp2 energy\n\n"

method_section_radical = "bq \"anion\"
clear
end
basis spherical
 O1 library O aug-cc-pvtz
 H1 library H aug-cc-pvtz
 O library aug-cc-pvdz
 H library aug-cc-pvdz
end

set lindep:n_dep 0

scf
vectors atomic output radical.movecs
 thresh 1d-6
 uhf
 doublet
 maxiter 350
 tol2e 1d-15
end

mp2
freeze atomic
end

task mp2 energy

set bq radical

scf
vectors input radical.movecs
 thresh 1d-6
 uhf
 doublet
 maxiter 350
 tol2e 1d-15
end

mp2
freeze atomic
end

task mp2 energy\n\n"

    mkpath("nwchem_input_files")
    @showprogress for i_sample in 1:num_samples
        if (
            !ispath(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, "_shell_positions_anion.txt")) || 
            !ispath(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, "_shell_positions_radical.txt"))
        )
            @warn "Couldn't open charges file for sample $i_sample. Moving on to the next sample."
            continue
        end
        _, cluster_labels, cluster_geom = read_xyz(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, ".xyz"))

        cluster_charge = sum([atom_charges[label] for label in cluster_labels[1]])
        
        # assumes OH is sorted to the top
        cluster_labels[1][1] = "O1"
        cluster_labels[1][2] = "H1"

        geom_string = geometry_to_string(cluster_geom[1], cluster_labels[1])
        open(string("nwchem_input_files/", nwchem_infile_prefix, "_sample_", i_sample, ".nw"), "w") do io
            # anion contribution to file
            write(io, header_string)
            write(io, string("charge ", cluster_charge, "\nGEOMETRY units angstrom noautoz nocenter\nsymmetry c1\n"))
            write(io, geom_string)
            write(io, "end\n\n")
            write(io, "bq \"anion\"\n")
            writedlm(io, readdlm(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, "_shell_positions_anion.txt")))
            write(io, "end\n\n")
            write(io, method_section_anion)
            
            # radical contribution to file
            write(io, string("charge ", cluster_charge+1, "\nGEOMETRY units angstrom noautoz nocenter\nsymmetry c1\n"))
            write(io, geom_string)
            write(io, "end\n\n")
            write(io, "bq \"radical\"\n")
            writedlm(io, readdlm(string("sampled_geoms_and_optimized_shells/cluster_sample_", i_sample, "_shell_positions_radical.txt")))
            write(io, "end\n\n")
            write(io, method_section_radical)
        end
    end
end
