include("nwchem_input_generator.jl")
using Combinatorics

"""
Writes a single Q-Chem input file.
"""
function write_input_file(infile_name::String, coords::Matrix{Float64}, labels::Vector{String}, rem_input::String, charge::Int, multiplicity::Int)
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

"""
Writes a Q-Chem input file for multiple jobs to be run sequentially.
rem_input is a file containing the rem block and any other blocks needed
for the desired calculation.
"""
function write_multi_input_file(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, rem_input::String, charge::Int, multiplicity::Int, read_rem_string_from_file::Bool=false)
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
            end_of_block = i_piece * 3000 < length(geoms) ? i_piece * 3000 : length(geoms)
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
    end
end

"""
 Writes a Q-Chem input file for multiple jobs to be run sequentially.
 rem_input is a file containing the rem block and any other blocks needed
 for the desired calculation.
 These jobs are specifically fragment-based calculations (usually for EDA or MBE).
 The fragments are specified with the fragment indices array and the corresponding
 charges and multiplicites of each fragment should be specified as well.
 """
function write_multi_input_file_fragments(
    infile_name::String,
    geoms::AbstractVector{Matrix{Float64}},
    labels::AbstractVector{Vector{String}},
    rem_input::String,
    charge::Int,
    multiplicity::Int,
    fragment_indices::Vector{Vector{Int}},
    fragment_charges::Vector{Int},
    fragment_multiplicities::Vector{Int},
    append::Bool=true
)
    mode = "a"
    if !append
        mode = "w"
    end
    rem_input_string = rem_input
    if isfile(infile_name) && append
        open(infile_name, mode) do io
            write(io, "\n@@@\n\n")
        end
    end
    open(infile_name, mode) do io
        for i in eachindex(geoms)
            write(io, "\$molecule\n")
            write(io, string(charge, " ", multiplicity, "\n"))
            write(io, "--\n")
            for i_frag in eachindex(fragment_indices)
                if fragment_indices[i_frag][1] <= size(geoms[i], 2)
                    write(io, string(fragment_charges[i_frag], " ", fragment_multiplicities[i_frag], "\n"))
                    geom_string = geometry_to_string(geoms[i][:, fragment_indices[i_frag]], labels[i][fragment_indices[i_frag]])
                    write(io, geom_string)
                    if i_frag != length(fragment_indices)
                        write(io, "--\n")
                    end
                else
                    @assert false "Provided fragment index which is larger than the number of indices in the geometry"
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

"""
Same as above but for a single geometry.
"""
function write_input_file_fragments(
    infile_name::String,
    coords::AbstractMatrix{Float64},
    labels::AbstractVector{String},
    rem_input::String,
    charge::Int,
    multiplicity::Int,
    fragment_indices::Vector{Vector{Int}},
    fragment_charges::Vector{Int},
    fragment_multiplicities::Vector{Int},
    append::Bool=true
)
    mode = "a"
    if !append
        mode = "w"
    end
    rem_input_string = rem_input
    if isfile(infile_name) && append
        open(infile_name, mode) do io
            write(io, "\n@@@\n\n")
        end
    end
    open(infile_name, mode) do io
        write(io, "\$molecule\n")
        write(io, string(charge, " ", multiplicity, "\n"))
        write(io, "--\n")
        for i_frag in eachindex(fragment_indices)
            if fragment_indices[i_frag][1] <= size(coords, 2)
                write(io, string(fragment_charges[i_frag], " ", fragment_multiplicities[i_frag], "\n"))
                geom_string = geometry_to_string(coords[:, fragment_indices[i_frag]], labels[fragment_indices[i_frag]])
                write(io, geom_string)
                if i_frag != length(fragment_indices)
                    write(io, "--\n")
                end
            else
                @assert false "Provided fragment index which is larger than the number of indices in the geometry"
            end
        end
        write(io, "\$end\n\n")
        write(io, rem_input_string)
    end
end

"""
Writes input files for arbitrary order of the MBE. All fragments of an order go in the same file.
Currently all fragments must be neutral and closed shell.
"""
function write_mbe_inputs(
    infile_name::String,
    coords::AbstractMatrix{Float64},
    labels::AbstractVector{String},
    rem_input::String,
    fragment_indices::AbstractVector{Vector{Int}},
    max_order::Int=2,
    include_full_system::Bool=true
)
    file_prefix = splitext(infile_name)[1]
    for i_mbe in 2:max_order
        for fragment_indices_combination in combinations(fragment_indices, i_mbe)
            flat_indices = reduce(vcat, fragment_indices_combination)
            @views subsystem_coords = coords[:, flat_indices]
            @views subsystem_labels = labels[flat_indices]
            shifted_indices = [zero(fragment_indices_combination[i]) for i in eachindex(fragment_indices_combination)]
            index = 1
            for i_frag in eachindex(shifted_indices)
                for i in eachindex(shifted_indices[i_frag])
                    shifted_indices[i_frag][i] = index
                    index += 1
                end
            end
            write_input_file_fragments(
                string(file_prefix, "_", i_mbe, "_body.in"),
                subsystem_coords,
                subsystem_labels,
                rem_input,
                0, 1,
                shifted_indices,
                zeros(Int, length(shifted_indices)),
                ones(Int, length(shifted_indices))
            )
        end
    end
    if include_full_system
        write_input_file_fragments(
            string(file_prefix, "_full_system.in"),
            coords,
            labels,
            rem_input,
            0, 1,
            fragment_indices,
            zeros(Int, length(fragment_indices)),
            ones(Int, length(fragment_indices))
        )
    end
end

function write_multi_input_file_fragments(
    infile_name::String,
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

"""
Takes a collection of geometries, labels, charges, and multiplicities
and generates a Q-Chem input file for each of them in a separate file.
The file are specified by an integer that goes in the file name.
The same rem input is used for all geometries.
"""
function write_separate_input_files_for_fragments(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, rem_input::String, fragment_charges::Vector{Int}, fragment_multiplicites::Vector{Int}, output_dir::Union{String, Nothing}=nothing)
    file_prefix = splitext(basename(infile_name))[1]
    dir_prefix = "."
    if output_dir !== nothing
        output_dir = strip(output_dir, '/')
        dir_prefix = output_dir
        if !isdir(output_dir)
            mkdir(output_dir)
        end
    end

    for i in eachindex(geoms)
        write_input_file(string(dir_prefix, "/", file_prefix, "_", i, ".in"), geoms[i], labels[i], rem_input, fragment_charges[i], fragment_multiplicites[i])
    end
end

"""
Same as above but for uncharged singlets.
"""
function write_separate_input_files_for_fragments(infile_name::String, geoms::AbstractVector{Matrix{Float64}}, labels::AbstractVector{Vector{String}}, rem_input::String, output_dir::Union{String, Nothing}=nothing)
    write_separate_input_files_for_fragments(infile_name, geoms, labels, rem_input, zeros(Int, length(geoms)), ones(Int, length(geoms)))
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

function multipole_field(x::Float64=0.0, y::Float64=0.0, z::Float64=0.0)
return "\n\$multipole_field
  X $x
  Y $y
  Z $z
\$end\n"
end

function get_fchk()
    return "\$rem
  JOBTYPE sp
  method wB97X-V
  BASIS def2-qzvppd
  SYMMETRY FALSE
  SYM_IGNORE TRUE
  IQMOL_FCHK true
\$end\n"
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
  mem_total  16000
  BASIS_LIN_DEP_THRESH 8
  THRESH 14
  SCF_CONVERGENCE 8
  SCF_PRINT_FRGM TRUE
\$end\n"
end

function bonded_eda_input()
    return "
\$rem
  jobtype sp
  gen_scfman true
  skip_gesman true
  EDA2 10                !! Go through the EDA2 code
  BONDED_EDA 2           !! Do the spin-projected bonded EDA
  EDA_POP_ANAL 0         !! No need to do population analysis on the fragments
  N_EDA_SPIN_FLIP 1      !! This is a 1-spin flip job
  EDA_CONTRACTION_ANAL 1 !! Breakdown POL term into rehybridization, contraction, pol
  exchange = wB97M-V          !! This can be DFT functionals
  basis def2-qzvppd
  THRESH 14
  mem_total 64000
  mem_static 2000
  max_scf_cycles 400
  SCF_CONVERGENCE 6
  mem_total 4000
  mem_static 1000
  ROSCF True
  scf_guess fragmo
  symmetry false
  sym_ignore true
  scf_algorithm gdm_ls
  scfmi_mode 1
  scf_print_frgm true
  frgm_method stoll
  frgm_lpcorr exact_scf
  basis_lin_dep_thresh 6
  scfmi_occs 0
  scfmi_virts 0
  child_mp true
  child_mp_orders 1233
  FRZ_ORTHO_DECOMP FALSE
  EDA_CLS_ELEC FALSE
  FRZ_RELAX true
  FRZ_RELAX_METHOD 2
  eda_pol_a False
\$end

\$rem_frgm
  scf_convergence 7
  scf_algorithm gdm_ls
  skip_gesman False
  SCF_GUESS sad
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

function wb97xv_qzvppd_rem()
    return "\$rem
  mem_total  16000
  ideriv                  1
  incdft                  0
  incfock                 0
  jobtype                 sp
  method                  wB97X-V
  unrestricted            false
  basis                   def2-qzvppd
  scf_algorithm           gdm
  scf_max_cycles          500
  scf_guess               sad
  scf_convergence         8
  thresh                  14
  symmetry                0
  sym_ignore              1
  gen_scfman              true
  gen_scfman_final        true
  internal_stability      false
  complex                 false
  chelpg true
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
