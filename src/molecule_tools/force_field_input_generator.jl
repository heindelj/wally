include("sample_configurations.jl")
include("qchem_input_generator.jl")
include("gdma_and_orient.jl")

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

    for i in (num_batches*num_jobs_per_batch+1):(num_batches*num_jobs_per_batch+num_leftover)
        generate_force_decomposition_input(
            string(infile_name_prefix, "_batch_", num_batches + 1, "_", i, ".in"),
            coords[i], labels[i], fragment_indices,
            charge, multiplicity,
            fragment_charges, fragment_multiplicities
        )
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