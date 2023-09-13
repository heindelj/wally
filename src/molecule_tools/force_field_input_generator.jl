include("sample_configurations.jl")
include("qchem_input_generator.jl")
include("gdma_and_orient.jl")
include("scans.jl")

function generate_all_inputs_main(
    infile_name::String,
    coords::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    fragment_indices::AbstractVector{Vector{Int}},
    charge::Int,
    multiplicity::Int,
    fragment_charges::Vector{Int},
    fragment_multiplicities::Vector{Int}
)
    # generate EDA input
    generate_fragment_inputs(
        string(infile_name, "_eda"), coords, labels,
        fragment_indices, eda_input(), 400,
        charge, multiplicity,
        fragment_charges, fragment_multiplicities,
        output_dir="eda_input"
    )
    # generate FDA input
    generate_fragment_inputs(
        string(infile_name, "_fda"), coords, labels,
        fragment_indices, fda_input(), 400,
        charge, multiplicity,
        fragment_charges, fragment_multiplicities,
        output_dir="fda_input"
    )
    write_fchk_input_files_multiple_fragmented_geoms(
        infile_name,
        coords, labels, get_fchk(), fragment_indices,
        fragment_charges, fragment_multiplicities
    )
    open("fchk_files/submit.slurm", "w") do io
        write(io, perlmutter_fchk_slurm_script_string())
    end
end

function generate_fragment_inputs(
    infile_name::String,
    coords::AbstractMatrix{Float64},
    labels::AbstractVector{String},
    fragment_indices::AbstractVector{Vector{Int}},
    rem_string::String=eda_input(),
    charge::Int=0,
    multiplicity::Int=1,
    fragment_charges::Union{Nothing, Vector{Int}}=nothing,
    fragment_multiplicities::Union{Nothing, Vector{Int}}=nothing;
    output_dir::Union{String, Nothing}=nothing
)
    if fragment_charges === nothing
        fragment_charges = zeros(Int, length(fragment_indices))
    end
    if fragment_multiplicities === nothing
        fragment_multiplicities = ones(Int, length(fragment_indices))
    end

    if output_dir !== nothing
        output_dir = strip(output_dir, '/')
        mkpath(output_dir)
        infile_name = string(output_dir, "/", infile_name)
    end

    write_input_file_fragments(
        infile_name,
        coords, labels, rem_string,
        charge, multiplicity,
        fragment_indices, fragment_charges, fragment_multiplicities
    )
end

function generate_fragment_inputs(
    infile_name_prefix::String,
    coords::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    fragment_indices::AbstractVector{Vector{Int}},
    rem_string::String=eda_input(),
    num_jobs_per_batch::Int=400,
    charge::Int=0,
    multiplicity::Int=1,
    fragment_charges::Union{Nothing,Vector{Int}}=nothing,
    fragment_multiplicities::Union{Nothing,Vector{Int}}=nothing;
    output_dir::Union{String,Nothing}=nothing
)
    # num_jobs_per_batch generates text files with the names of
    # generated input files up to that number. This is useful for
    # making the batch script that submits the inputs to a cluster.

    # make the output dir and update file name
    # so we don't pass the output dir to the function
    # which makes the file.
    if output_dir !== nothing
        output_dir = strip(output_dir, '/')
        mkpath(output_dir)
        infile_name_prefix = string(output_dir, "/", infile_name_prefix)
    else
        output_dir = ""
    end

    num_batches = length(coords) ÷ num_jobs_per_batch
    num_leftover = length(coords) - num_jobs_per_batch * num_batches
    for i_batch in 1:num_batches
        batch_file_name = string("batch_", i_batch, ".slurm")
        if output_dir !== nothing
            batch_file_name = string(output_dir, "/batch_", i_batch, ".slurm")
        end
        open(batch_file_name, "w") do io
            write(io, perlmutter_slurm_script_string(lstrip(replace(infile_name_prefix, output_dir => ""), '/'), i_batch, (i_batch - 1) * num_jobs_per_batch + 1, num_jobs_per_batch))
        end

        for i in ((i_batch-1)*num_jobs_per_batch+1):(i_batch*num_jobs_per_batch)
            generate_fragment_inputs(
                string(infile_name_prefix, "_batch_", i_batch, "_", i, ".in"),
                coords[i], labels[i], fragment_indices, rem_string,
                charge, multiplicity,
                fragment_charges, fragment_multiplicities
            )
        end
    end

    if num_leftover == 0
        return
    end

    # deal with the leftover structures
    batch_file_name = string("batch_", num_batches + 1, ".slurm")
    if output_dir !== nothing
        batch_file_name = string(output_dir, "/batch_", num_batches + 1, ".slurm")
    end
    open(batch_file_name, "w") do io
        write(io, perlmutter_slurm_script_string(lstrip(replace(infile_name_prefix, output_dir => ""), '/'), num_batches + 1, num_batches * num_jobs_per_batch + 1, num_leftover))
    end

    #for i in (num_batches*num_jobs_per_batch+1):(num_batches*num_jobs_per_batch+num_leftover)
    #    generate_force_decomposition_input(
    #        string(infile_name_prefix, "_batch_", num_batches + 1, "_", i, ".in"),
    #        coords[i], labels[i], fragment_indices,
    #        charge, multiplicity,
    #        fragment_charges, fragment_multiplicities
    #    )
    #end
end

initial_distances = Dict(
        :LiF => 1.5740842692,
        :LiCl => 2.0273134221999998,
        :LiBr => 2.1804909851999996,
        :LiI => 2.3928627104,
        :NaF => 1.9395644696,
        :NaCl => 2.3681930533999997,
        :NaBr => 2.5125576179999998,
        :NaI => 2.7130080771999996,
        :KF => 2.191804663,
        :KCl => 2.6883189056,
        :KBr => 2.8474846406000003,
        :KI => 3.0659928548,
        :RbF => 2.3418174634,
        :RbCl => 2.848191689,
        :RbBr => 3.0101616448,
        :RbI => 3.2302740466,
        :CsF => 2.3791859333999996,
        :CsCl => 2.9408768782,
        :CsBr => 3.1139594843999996,
        :CsI => 3.3471488736,
        :MgF => 1.6965971996000002,
        :MgCl => 2.1010917015999997,
        :MgBr => 2.2447553162,
        :MgI => 2.4449346346,
        :CaF => 1.8660764668,
        :CaCl => 2.3192035096000003,
        :CaBr => 2.4710069642,
        :CaI => 2.6804184199999996,

        :Li2F =>1.675469,
        :Na2F => 2.021674,
        :K2F => 2.348659,
        :Rb2F => 2.501305,
        :Cs2F => 2.607304,
        :Mg2F => 1.922883,
        :Ca2F => 2.197071,
        :LiF2 =>1.697896,
        :NaF2 => 2.054501,
        :KF2 => 2.400956,
        :RbF2 => 2.557590,
        :CsF2 => 2.673066,
        :MgF2 => 1.754368,
        :CaF2 => 2.027367,

        :Li2Cl =>2.142979,
        :Na2Cl => 2.474615,
        :K2Cl => 2.837262,
        :Rb2Cl => 2.996256,
        :Cs2Cl => 3.134003,
        :Mg2Cl => 2.346741,
        :Ca2Cl => 2.652802,
        :LiCl2 =>2.161592,
        :NaCl2 => 2.496780,
        :KCl2 => 2.878317,
        :RbCl2 => 3.046327,
        :CsCl2 => 3.204202,
        :MgCl2 => 2.172486,
        :CaCl2 => 2.469349,

        :Li2Br =>2.297300,
        :Na2Br => 2.624763,
        :K2Br => 2.992072,
        :Rb2Br => 3.154446,
        :Cs2Br => 3.292489,
        :Mg2Br => 2.487264,
        :Ca2Br => 2.799551,
        :LiBr2 =>2.321164,
        :NaBr2 => 2.649016,
        :KBr2 => 3.033684,
        :RbBr2 => 3.204445,
        :CsBr2 => 3.364454,
        :MgBr2 => 2.323007,
        :CaBr2 => 2.619850,

        :Li2I =>2.508365,
        :Na2I => 2.832592,
        :K2I => 3.208172,
        :Rb2I => 3.375760,
        :Cs2I => 3.527121,
        :Mg2I => 2.684221,
        :Ca2I => 2.999745,
        :LiI2 =>2.539843,
        :NaI2 => 2.862934,
        :KI2 => 3.252194,
        :RbI2 => 3.426597,
        :CsI2 => 3.591727,
        :MgI2 => 2.531817,
        :CaI2 => 2.830109,
        

        # these are just guesses since there is no minimum
        :FF => 2.5,
        :ClCl => 3.7,
        :BrBr => 4.0,
        :II => 4.6,
        :LiLi => 2.4,
        :NaNa => 2.82,
        :KK => 3.6,
        :RbRb => 3.8,
        :CsCs => 4.2,
        :MgMg => 2.3,
        :CaCa => 2.8,
    )

function generate_ion_pair_scan_inputs()
    anions = ["F", "Cl", "Br", "I"]
    monovalent_cations = ["Li", "Na", "K", "Rb", "Cs"]
    divalent_cations = ["Mg", "Ca"]

    mkpath("ion_pair_scans")
    # same ion monovalent cations
    #for cation in monovalent_cations
    #    geom = zeros(3, 2)
    #    geom[1, 2] = initial_distances[Symbol(cation, cation)]
    #    scan_geoms = distance_scan(geom, (1, 2), -0.6, 5.4, 51)
    #    write_multi_input_file_fragments(
    #        string("ion_pair_scans/", cation, "_", cation, "_scan_wb97xv_eda.in"),
    #        scan_geoms, [cation, cation], eda_input(),
    #        2, 1, [[1], [2]], [1, 1], [1, 1]
    #    )
    #end

    # same ion divalent cations
    #for cation in divalent_cations
    #    geom = zeros(3, 2)
    #    geom[1, 2] = initial_distances[Symbol(cation, cation)]
    #    scan_geoms = distance_scan(geom, (1, 2), -0.6, 5.4, 61)
    #    write_multi_input_file_fragments(
    #        string("ion_pair_scans/", cation, "_", cation, "_scan_wb97xv_eda.in"),
    #        scan_geoms, [cation, cation], eda_input(),
    #        4, 1, [[1], [2]], [2, 2], [1, 1]
    #    )
    #end

    # same ion anions
    #for anion in anions
    #    geom = zeros(3, 2)
    #    geom[1, 2] = initial_distances[Symbol(anion, anion)]
    #    scan_geoms = distance_scan(geom, (1, 2), -0.6, 5.4, 61)
    #    write_multi_input_file_fragments(
    #        string("ion_pair_scans/", anion, "_", anion, "_scan_wb97xv_eda.in"),
    #        scan_geoms, [anion, anion], eda_input(),
    #        -2, 1, [[1], [2]], [-1, -1], [1, 1]
    #    )
    #end

    # anion monovalent cation
    #for cation in monovalent_cations
    #    for anion in anions
    #        geom = zeros(3, 2)
    #        geom[1, 2] = initial_distances[Symbol(cation, anion)]
    #        scan_geoms = distance_scan(geom, (1, 2), -0.6, 5.4, 61)
    #        write_multi_input_file_fragments(
    #            string("ion_pair_scans/", cation, "_", anion, "_scan_wb97xv_eda.in"),
    #            scan_geoms, [cation, anion], eda_input(),
    #            0, 1, [[1], [2]], [1, -1], [1, 1]
    #        )
    #    end
    #end

    # anion divalent cation
    #for cation in divalent_cations
    #    for anion in anions
    #        geom = zeros(3, 2)
    #        geom[1, 2] = initial_distances[Symbol(cation, anion)]
    #        scan_geoms = distance_scan(geom, (1, 2), -0.6, 5.4, 61)
    #        write_multi_input_file_fragments(
    #            string("ion_pair_scans/", cation, "_", anion, "_scan_wb97xv_eda.in"),
    #            scan_geoms, [cation, anion], eda_input(),
    #            1, 1, [[1], [2]], [2, -1], [1, 1]
    #        )
    #    end
    #end

    # monovalent cation divalent cation
    #for cation_1 in monovalent_cations
    #    for cation_2 in divalent_cations
    #        geom = zeros(3, 2)
    #        geom[1, 2] = 0.5 * (initial_distances[Symbol(cation_1, cation_1)] + initial_distances[Symbol(cation_2, cation_2)])
    #        scan_geoms = distance_scan(geom, (1, 2), -0.6, 5.4, 61)
    #        write_multi_input_file_fragments(
    #            string("ion_pair_scans/", cation_1, "_", cation_2, "_scan_wb97xv_eda.in"),
    #            scan_geoms, [cation_1, cation_2], eda_input(),
    #            3, 1, [[1], [2]], [1, 2], [1, 1]
    #        )
    #    end
    #end

    # monovalent cation monovalent cation
    for (i, cation_1) in enumerate(monovalent_cations)
        for (j, cation_2) in enumerate(monovalent_cations)
            if i < j
                geom = zeros(3, 2)
                geom[1, 2] = 0.5 * (initial_distances[Symbol(cation_1, cation_1)] + initial_distances[Symbol(cation_2, cation_2)])
                scan_geoms = distance_scan(geom, (1, 2), -0.6, 5.4, 61)
                write_multi_input_file_fragments(
                    string("ion_pair_scans/", cation_1, "_", cation_2, "_scan_wb97xv_eda.in"),
                    scan_geoms, [cation_1, cation_2], eda_input(),
                    2, 1, [[1], [2]], [1, 1], [1, 1]
                )
            end
        end
    end
end

function generate_ion_triples_2d_scan_inputs()
    anions = ["F", "Cl", "Br", "I"]
    monovalent_cations = ["Li", "Na", "K", "Rb", "Cs"]
    divalent_cations = ["Mg", "Ca"]

    mkpath("ion_triple_scans")
    # two anion one monovalent cation
    for cation in monovalent_cations
        for anion in anions
            labels = [anion, cation, anion]
            geom = zeros(3, 3)
            initial_pos = initial_distances[Symbol(cation, anion, :2)]
            geom[1, 1] = -initial_pos
            geom[1, 3] =  initial_pos
            distance_scan_geoms = distance_scan(geom, (1, 3), -0.6, 5.4, 31)
            scan_geoms = angle_scan.(distance_scan_geoms, ((1,2,3),), (0.0,), (90.0,), (10,))
            scan_geoms = reduce(vcat, scan_geoms)
            write_multi_input_file_fragments(
                string("ion_triple_scans/", cation, "_", anion, "2_2d_scan_wb97xv_eda.in"),
                scan_geoms, labels, eda_input(),
                -1, 1, [[1], [2], [3]], [-1, 1, -1], [1, 1, 1]
            )
        end
    end

    # two anion one divalent cation
    for cation in divalent_cations
        for anion in anions
            labels = [anion, cation, anion]
            geom = zeros(3, 3)
            initial_pos = initial_distances[Symbol(cation, anion, :2)]
            geom[1, 1] = -initial_pos
            geom[1, 3] =  initial_pos
            distance_scan_geoms = distance_scan(geom, (1, 3), -0.6, 5.4, 31)
            scan_geoms = angle_scan.(distance_scan_geoms, ((1,2,3),), (0.0,), (90.0,), (10,))
            scan_geoms = reduce(vcat, scan_geoms)
            write_multi_input_file_fragments(
                string("ion_triple_scans/", cation, "_", anion, "2_2d_scan_wb97xv_eda.in"),
                scan_geoms, labels, eda_input(),
                0, 1, [[1], [2], [3]], [-1, 2, -1], [1, 1, 1]
            )
        end
    end

    # two monovalent cation one anion
    for cation in monovalent_cations
        for anion in anions
            labels = [cation, anion, cation]
            geom = zeros(3, 3)
            initial_pos = initial_distances[Symbol(cation, :2, anion)]
            geom[1, 1] = -initial_pos
            geom[1, 3] =  initial_pos
            distance_scan_geoms = distance_scan(geom, (1, 3), -0.6, 5.4, 31)
            scan_geoms = angle_scan.(distance_scan_geoms, ((1,2,3),), (0.0,), (90.0,), (10,))
            scan_geoms = reduce(vcat, scan_geoms)
            write_multi_input_file_fragments(
                string("ion_triple_scans/", cation, "2_", anion, "_2d_scan_wb97xv_eda.in"),
                scan_geoms, labels, eda_input(),
                1, 1, [[1], [2], [3]], [1, -1, 1], [1, 1, 1]
            )
        end
    end

    # two divalent cation one anion
    for cation in divalent_cations
        for anion in anions
            labels = [cation, anion, cation]
            geom = zeros(3, 3)
            initial_pos = initial_distances[Symbol(cation, :2, anion)]
            geom[1, 1] = -initial_pos
            geom[1, 3] =  initial_pos
            distance_scan_geoms = distance_scan(geom, (1, 3), -0.6, 5.4, 31)
            scan_geoms = angle_scan.(distance_scan_geoms, ((1,2,3),), (0.0,), (90.0,), (10,))
            scan_geoms = reduce(vcat, scan_geoms)
            write_multi_input_file_fragments(
                string("ion_triple_scans/", cation, "2_", anion, "_2d_scan_wb97xv_eda.in"),
                scan_geoms, labels, eda_input(),
                3, 1, [[1], [2], [3]], [2, -1, 2], [1, 1, 1]
            )
        end
    end
end

function generate_ion_triples_optimization_input()
    anions = ["F", "Cl", "Br", "I"]
    monovalent_cations = ["Li", "Na", "K", "Rb", "Cs"]
    divalent_cations = ["Mg", "Ca"]

    mkpath("ion_triples_optimization")
    # two anion one monovalent cation
    for cation in monovalent_cations
        for anion in anions
            labels = [anion, cation, anion]
            geom = zeros(3, 3)
            initial_pos = 0.5 * (initial_distances[Symbol(cation, anion)] - 0.2)
            geom[1, 1] = -initial_pos
            geom[1, 3] =  initial_pos
            write_input_file(
                string("ion_triples_optimization/", cation, "_", anion, "2_wb97xv_tzvppd_opt.in"),
                geom, labels, wb97xv_tzvppd_opt(),
                -1, 1
            )
        end
    end

    # two anion one divalent cation
    for cation in divalent_cations
        for anion in anions
            labels = [anion, cation, anion]
            geom = zeros(3, 3)
            initial_pos = 0.5 * (initial_distances[Symbol(cation, anion)] - 0.2)
            geom[1, 1] = -initial_pos
            geom[1, 3] =  initial_pos
            write_input_file(
                string("ion_triples_optimization/", cation, "_", anion, "2_wb97xv_tzvppd_opt.in"),
                geom, labels, wb97xv_tzvppd_opt(),
                0, 1
            )
        end
    end

    # two monovalent cation one anion
    for cation in monovalent_cations
        for anion in anions
            labels = [cation, anion, cation]
            geom = zeros(3, 3)
            initial_pos = 0.5 * (initial_distances[Symbol(cation, anion)] - 0.2)
            geom[1, 1] = -initial_pos
            geom[1, 3] =  initial_pos
            write_input_file(
                string("ion_triples_optimization/", cation, "2_", anion, "_wb97xv_tzvppd_opt.in"),
                geom, labels, wb97xv_tzvppd_opt(),
                1, 1
            )
        end
    end

    # two divalent cation one anion
    for cation in divalent_cations
        for anion in anions
            labels = [cation, anion, cation]
            geom = zeros(3, 3)
            initial_pos = 0.5 * (initial_distances[Symbol(cation, anion)] - 0.2)
            geom[1, 1] = -initial_pos
            geom[1, 3] =  initial_pos
            write_input_file(
                string("ion_triples_optimization/", cation, "2_", anion, "_wb97xv_tzvppd_opt.in"),
                geom, labels, wb97xv_tzvppd_opt(),
                3, 1
            )
        end
    end
end

function perlmutter_slurm_script_string(infile_prefix::AbstractString, batch_number::Int, index_start::Int, num_per_batch::Int)
    index_end = index_start + num_per_batch - 1
    full_file = string(infile_prefix, "_batch_", batch_number)
    return string("#!/bin/bash
#SBATCH -A m2101
#SBATCH -t 12:00:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -e batch_job_$batch_number.error
#SBATCH -J batch_job_$batch_number
#SBATCH -o batch_job_$batch_number.stdout
cd \$SLURM_SUBMIT_DIR
export SCRDIR=\$SCRATCH/scr.\$SLURM_JOBID
#export QC=/global/cfs/projectdirs/m2101/heindelj/qchem-trunk
#export QCAUX=/global/cfs/projectdirs/m2101/heindelj/qcaux
#export QCRSH=ssh
#export QCMPI=openmpi
#export PATH=\$PATH:\$QC/bin:\$QC/bin/perl
mkdir -p \$SCRDIR
export QCSCRATCH=\$SCRDIR
module load qchem
env > batch_job_$batch_number.output.\$SLURM_JOBID.\$SLURM_NNODES  2>&1
let j=1
for((i=$index_start;i<=$index_end;i+=4));
do
let j=i\n",
string("qchem -save -nt 32 $full_file", "_\$j.in ", "$full_file", "_\$j.out &\n"),
string("let j=j+1\n"),
string("qchem -save -nt 32 $full_file", "_\$j.in ", "$full_file", "_\$j.out &\n"),
string("let j=j+1\n"),
string("qchem -save -nt 32 $full_file", "_\$j.in ", "$full_file", "_\$j.out &\n"),
string("let j=j+1\n"),
string("qchem -save -nt 32 $full_file", "_\$j.in ", "$full_file", "_\$j.out &\n"),
"wait\n",
"done"
)
end

function perlmutter_fchk_slurm_script_string()
    return string("#!/bin/bash
#SBATCH -A m2101
#SBATCH -t 12:00:00
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -e fchk_job.error
#SBATCH -J fchk_job
#SBATCH -o fchk_job.stdout
cd \$SLURM_SUBMIT_DIR
export SCRDIR=\$SCRATCH/scr.\$SLURM_JOBID
export QCAUX=/global/cfs/projectdirs/m2101/heindelj/qcaux
mkdir -p \$SCRDIR
export QCSCRATCH=\$SCRDIR
module load qchem
env > fchk_job.output.\$SLURM_JOBID.\$SLURM_NNODES  2>&1
ls *.in > infiles.txt
sed -i 's/.in//' infiles.txt
while read line;
do
qchem -save -nt 64 \$line.in \$line.out
done < infiles.txt"
)
end