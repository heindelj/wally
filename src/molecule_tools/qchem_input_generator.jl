include("nwchem_input_generator.jl")

function write_input_file(infile_name::String, coords::Matrix{Float64}, labels::Vector{String}, rem_input::String, charge::Int, multiplicity::Int)
    """
    Writes a single Q-Chem input file.
    """
    used_input_name = next_unique_name(infile_name)
    rem_input_string = rem_input
    geom_string = geometry_to_string(coords, labels)
    open(used_input_name, "w") do io
        write(io, "\$molecule\n")
        write(io, string(charge, " ", multiplicity, "\n"))
        write(io, geom_string)
        write(io, "\$end\n\n")
        write(io, rem_input_string)
    end
    return used_input_name
end

function write_multi_input_file(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, rem_input::String, charge::Int, multiplicity::Int, read_rem_string_from_file::Bool=false)
    """
    Writes a Q-Chem input file for multiple jobs to be run sequentially.
    rem_input is a file containing the rem block and any other blocks needed
    for the desired calculation.
    """ 
    used_input_name = next_unique_name(infile_name)
    rem_input_string = rem_input
    if read_rem_string_from_file
        rem_input_string = read(rem_input, String)
    end # otherwise rem string is just provided directly

    if length(geoms) > 3000
        num_pieces = (length(geoms) ÷ 3000) + 1
        names_to_write = String[]
        names_written = 1
        for i_piece in 1:num_pieces
            used_input_name_split = string(splitext(used_input_name)[1], "_part_", i_piece, ".in")
            push!(names_to_write, used_input_name_split)
            end_of_block = i_piece*3000 < length(geoms) ? i_piece*3000 : length(geoms)
            geom_block_indices = ((i_piece-1)*3000+1):end_of_block
            open(used_input_name_split, "w") do io
                for i in geom_block_indices
                    geom_string = geometry_to_string(geoms[i], labels[i])
                    write(io, "\$molecule\n")
                    write(io, string(charge, " ", multiplicity, "\n"))
                    write(io, geom_string)
                    write(io, "\$end\n\n")
                    write(io, rem_input_string)
                    if i != end_of_block
                        write(io, "\n@@@\n\n")
                    end
                end
            end
            if (i_piece % 3) == 0 || i_piece == num_pieces
                open(string(splitext(infile_name)[1], "_part_", names_written, ".slurm"), "w") do io
                    slurm_string = write_lawrencium_slurm_scripts(names_to_write)
                    write(io, slurm_string)
                end
                names_written += 1
                empty!(names_to_write)
            end
        end
    else
        open(used_input_name, "w") do io
            for i in eachindex(geoms)
                geom_string = geometry_to_string(geoms[i], labels[i])
                write(io, "\$molecule\n")
                write(io, string(charge, " ", multiplicity, "\n"))
                write(io, geom_string)
                write(io, "\$end\n\n")
                write(io, rem_input_string)
                if i != length(geoms)
                    write(io, "\n@@@\n\n")
                end
            end
        end
        open(string(splitext(infile_name)[1], ".slurm"), "w") do io
            slurm_string = write_lawrencium_slurm_scripts([used_input_name])
            write(io, slurm_string)
        end
    end
end

function write_multi_input_file_fragments(
    infile_name::String,
    geoms::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    rem_input::String,
    charge::Int,
    multiplicity::Int,
    fragment_indices::Vector{Vector{Int}},
    fragment_charges::Vector{Int},
    fragment_multiplicities::Vector{Int}
    )
    """
    Writes a Q-Chem input file for multiple jobs to be run sequentially.
    rem_input is a file containing the rem block and any other blocks needed
    for the desired calculation.
    These jobs are specifically fragment-based calculations (usually for EDA or MBE).
    The fragments are specified with the fragment indices array and the corresponding
    charges and multiplicites of each fragment should be specified as well.
    """ 
    rem_input_string = rem_input
    open(infile_name, "w") do io
        for i in eachindex(geoms)
            write(io, "\$molecule\n")
            write(io, string(charge, " ", multiplicity, "\n"))
            write(io, "--\n")
            for i_frag in eachindex(fragment_indices)
                if fragment_indices[i_frag][1] < size(geoms[i], 2)
                    write(io, string(fragment_charges[i_frag], " ", fragment_multiplicities[i_frag], "\n"))
                    geom_string = geometry_to_string(geoms[i][:, fragment_indices[i_frag]], labels[i][fragment_indices[i_frag]])
                    write(io, geom_string)
                    if i_frag != length(fragment_indices)
                        write(io, "--\n")
                    end
                end
            end
            write(io, "\$end\n\n")
            write(io, rem_input_string)
            if i != length(geoms)
                write(io, "\n@@@\n\n")
            end
        end
    end
end

function write_multi_input_file_fragments(infile_name::String,
    geoms::AbstractVector{Matrix{Float64}},
    labels::Vector{String},
    rem_input::String,
    charge::Int,
    multiplicity::Int,
    fragment_indices::Vector{Vector{Int}},
    fragment_charges::Vector{Int},
    fragment_multiplicities::Vector{Int}
    )
    write_multi_input_file_fragments(infile_name, geoms, [labels for _ in eachindex(geoms)], rem_input, charge, multiplicity, fragment_indices, fragment_charges, fragment_multiplicities)
end

function write_multi_input_file(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::Vector{String}, rem_input::String, charge::Int, multiplicity::Int, read_rem_string_from_file::Bool=false)
    write_multi_input_file(infile_name, geoms, [labels for _ in eachindex(geoms)], rem_input, charge, multiplicity, read_rem_string_from_file)
end

function make_qchem_job(infile_name::String, coords::Matrix{Float64}, labels::Vector{String}, rem_input::String, charge::Int, multiplicity::Int)
    """
    Takes the appropriate data and arguments to construct and run a Q-Chem job.
    Returns a functor which can then be used to run the job whenever it is
    convenient for the callee. The functor is constructed by a call to run_qchem_job.
    Specifically, these can be packed into a jobs array and run with a one_to_many call.
    """
    return () -> run_qchem_job(write_input_file(infile_name, coords, labels, rem_input, charge, multiplicity))
end

function make_qchem_job(infile_name::String)
    """
    Constructs a Q-Chem job from an already assembled input file and returns as functor.
    This code path assumes the output file name where we remove the
    .in suffix will be a valid output name. That is, spawning multiple
    jobs from the same input file will result in the output files
    being overwritten. In that case, copy the input file to a different
    name and run separate jobs with the identical inputs.
    """
    return () -> run_qchem_job(infile_name)
end

function run_qchem_job(file_path::String, nt::Int=32)
    out_name = string(splitext(file_path)[1], ".out")
    run(pipeline(`'qchem' '-save' '-nt' $(nt) $(file_path) $(out_name) '&'`))
end

function eda_input()
    return "\$rem
  JOBTYPE EDA
  EDA2 1
  method wB97X-V
  BASIS def2-qzvppd
  XC_GRID 000099000590
  NL_GRID 1
  UNRESTRICTED FALSE
  MAX_SCF_CYCLES 200
  SYMMETRY FALSE
  SYM_IGNORE TRUE
  MEM_STATIC 2000
  BASIS_LIN_DEP_THRESH 8
  THRESH 14
  SCF_CONVERGENCE 8
  SCF_PRINT_FRGM TRUE
\$end\n"
end

function constrained_opt_input()
    return "\$rem
  JOBTYPE opt
  method wB97X-V
  BASIS def2-qzvppd
  XC_GRID 000099000590
  NL_GRID 1
  UNRESTRICTED FALSE
  MAX_SCF_CYCLES 200
  SYMMETRY FALSE
  SYM_IGNORE TRUE
  MEM_STATIC 2000
  BASIS_LIN_DEP_THRESH 8
  THRESH 14
  SCF_CONVERGENCE 8
\$end\n

\$opt
fixed
1 xyz
4 xyz
7 xyz
endfixed
\$end
"
end

function wb97xv_tzvppd_with_nbo_rem()
    return "\$rem
  mem_total  16000
  ideriv                  1
  incdft                  0
  incfock                 0
  jobtype                 sp
  method                  wB97X-V
  unrestricted            false
  basis                   def2-tzvppd
  scf_algorithm           gdm
  scf_max_cycles          500
  scf_guess               sad
  scf_convergence         8
  thresh                  14
  symmetry                0
  sym_ignore              1
  nbo                     1
  gen_scfman              true
  gen_scfman_final        true
  internal_stability      false
  complex                 false
  chelpg true
\$end

\$nbo
  BNDIDX
\$end\n"
end

function wb97mv_avtz_with_molden_rem()
    return "\$rem
  jobtype                 sp
  method                  wB97M-V
  unrestricted            false
  basis                   aug-cc-PVTZ
  xc_grid        2
  scf_algorithm           gdm
  scf_max_cycles          500
  scf_guess               sad
  scf_convergence         8
  thresh                  14
  symmetry                0
  sym_ignore              1
  molden_format           true
  print_orbitals          true
\$end\n"
end

function wb97mv_avtz_with_molden_and_nbo_rem()
    return "\$rem
  jobtype                 sp
  method                  wB97M-V
  unrestricted            false
  basis                   aug-cc-PVTZ
  xc_grid        2
  scf_algorithm           gdm
  scf_max_cycles          500
  scf_guess               sad
  scf_convergence         7
  thresh                  14
  symmetry                0
  sym_ignore              1
  molden_format           true
  print_orbitals          true
  nbo                     1
\$end\n

\$nbo
  BNDIDX
\$end\n"
end

function get_molden_file_from_qchem_output(infile::String)
    """
    Extracts molden files from a qchem output file. If there is
    more than one, then all molden files are extracted and written
    with a numeric suffix.
    """
    all_molden_inputs = String[]
    lines = readlines(infile, keep=true)
    out_string = ""
    reading_molden_block = false
    for line in lines
        if occursin("END", line) == true && reading_molden_block == true
            reading_molden_block = false
            push!(all_molden_inputs, out_string)
            out_string = ""
        elseif occursin("FORMATTED INPUT FILE", line)
            reading_molden_block = true
        elseif reading_molden_block
            out_string = string(out_string, line) # terribly inefficient
        end
    end
    return all_molden_inputs
end

function write_lawrencium_slurm_scripts(infiles::Vector{String})
    infile_1_prefix = splitext(infiles[1])[1]
    qchem_commands = [string("qchem -save -nt 32 ", infile, " ", splitext(infile)[1], ".out") for infile in infiles]

    slurm_options = "#!/bin/bash
#SBATCH --account lr_ninjaone
#SBATCH --time 72:00:00
#SBATCH --partition csd_lr6_192
#SBATCH --qos=condo_ninjaone
#SBATCH -N 1
#SBATCH -e $infile_1_prefix.error
#SBATCH -J $infile_1_prefix
#SBATCH -o $infile_1_prefix.stdout
cd \$SLURM_SUBMIT_DIR
export QCSRC=/global/scratch/users/nancy_guan/qchem_src/qchem_601
export QC=\$QCSRC
export QCREF=\$QCSRC/qcref
export QCAUX=\$QCSRC/qcaux
export QCPROG=\$QC/build/qcprog.exe
export PATH=\$QC/bin:\$QC/bin/perl:\$PATH
module load cmake/3.17.0 gcc/7.4.0 mkl boost python/3.7
export QCTHREADS=32
export OMP_NUM_THREADS=32
mkdir -p /global/scratch/users/\$USER/scratch_space/scr.\$SLURM_JOBID
export QCSCRATCH=/global/scratch/users/\$USER/scratch_space/scr.\$SLURM_JOBID
env > $infile_1_prefix.output.\$SLURM_JOBID.\$SLURM_NNODES  2>&1"

    for qchem_command in qchem_commands
        slurm_options = string(slurm_options, "\n", qchem_command)
    end

    return slurm_options
end