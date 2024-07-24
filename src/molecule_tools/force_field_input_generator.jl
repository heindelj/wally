include("sample_configurations.jl")
include("qchem_input_generator.jl")
include("gdma_and_orient.jl")
include("scans.jl")
using Graphs, Random

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

function generate_ion_water_cluster_optimization_inputs(file_prefix::String, geoms::Vector{Matrix{Float64}}, labels::Vector{Vector{String}}, num_geoms_to_opt::Int=10)
    # always store the first graph since it is the lowest in energy
    unique_geom_indices = Int[]
    graphs = MolecularGraph[]
    G = build_noncovalent_molecular_graph(geoms[1], labels[1])
    push!(graphs, G)
    push!(unique_geom_indices, 1)

    for i in eachindex(geoms)
        G = build_noncovalent_molecular_graph(geoms[i], labels[i])
        num_isomorphs = 0
        for i_graph in eachindex(graphs)
            num_isomorphs += Graphs.Experimental.count_isomorph(G.G, graphs[i_graph].G)
        end
        if num_isomorphs == 0
            push!(unique_geom_indices, i)
            push!(graphs, G)
        end
    end

    if length(graphs) < num_geoms_to_opt
        num_geoms_to_opt = length(graphs)
    end

    mkpath(file_prefix)
    charge = get_total_charge(labels[1])
    for i in 1:num_geoms_to_opt
        write_input_file(
            string(file_prefix, "/", file_prefix, "_wb97xv_tzvppd_opt_", i, ".in"),
            geoms[unique_geom_indices[i]], labels[unique_geom_indices[i]], wb97xv_tzvppd_opt(),
            charge, 1
        )
    end
end

"""
Samples nmers of a particular size and composition and generates EDA input files.
Many input files will be made depending on how many nmers are sampled and the
batch size given as an option. Default number of EDA calculations per input file is 100.
"""
function sample_nmers_and_write_eda_input_files(
    output_file_prefix::String, xyz_file::String,
    num_fragments_in_nmer::Int, desired_fragment_labels::Vector{String},
    charge::Int, multiplicity::Int, fragment_indices::Vector{Vector{Int}},
    always_take_n_most_compact::Int=2, randomly_select_n_remaining::Int=3,
    num_nmers_per_input_file::Int=2, num_geometries_used::Int=20
)
    _, labels, geoms = read_xyz(xyz_file)
    geom_indices = randperm(length(geoms))
    if length(geom_indices) > num_geometries_used
        geom_indices = geom_indices[1:20]
    end
    rand_geoms = geoms[geom_indices]
    rand_labels = labels[geom_indices]
    all_nmer_labels = Vector{String}[]
    all_nmer_geoms = Matrix{Float64}[]
    for i in eachindex(rand_geoms)
        nmer_labels, nmer_geoms = sample_nmers_from_cluster_by_formula(rand_geoms[i], rand_labels[i], num_fragments_in_nmer, desired_fragment_labels)
        if length(nmer_labels) > always_take_n_most_compact
            append!(all_nmer_labels, nmer_labels[1:always_take_n_most_compact])
            append!(all_nmer_geoms, nmer_geoms[1:always_take_n_most_compact])
        else
            append!(all_nmer_labels, nmer_labels)
            append!(all_nmer_geoms, nmer_geoms)
        end
        if (length(nmer_labels) - always_take_n_most_compact) >= randomly_select_n_remaining
            indices = [(always_take_n_most_compact+1):length(nmer_labels)...]
            if length(indices) < randomly_select_n_remaining
                append!(all_nmer_labels, nmer_labels[indices])
                append!(all_nmer_geoms, nmer_geoms[indices])
            else
                selected_indices = indices[randperm(length(indices))[1:randomly_select_n_remaining]]
                append!(all_nmer_labels, nmer_labels[selected_indices])
                append!(all_nmer_geoms, nmer_geoms[selected_indices])
            end
        end
    end
    if length(all_nmer_geoms) == 0
        @assert false "Failed to find any matching nmers!"
    end
    num_files = ((length(all_nmer_geoms)-1) ÷ num_nmers_per_input_file) + 1
    for i_file in 1:num_files
        last_batch_index = i_file*num_nmers_per_input_file < length(all_nmer_geoms) ? i_file*num_nmers_per_input_file : length(all_nmer_geoms)
        fragment_charges = [get_total_charge(all_nmer_labels[1][fragment_indices[i]]) for i in eachindex(fragment_indices)]
        fragment_multiplicities = ones(Int, num_fragments_in_nmer)
        write_multi_input_file_fragments(
            string(output_file_prefix, "_", i_file, ".in"),
            all_nmer_geoms[((i_file-1)*num_nmers_per_input_file+1):last_batch_index],
            all_nmer_labels[((i_file-1)*num_nmers_per_input_file+1):last_batch_index],
            eda_input(), charge, multiplicity,
            fragment_indices, fragment_charges, fragment_multiplicities
        )
    end
end

function write_two_anion_one_cation_scans()
    anion_labels=["F", "Cl", "Br", "I"]
    cation_labels=["Li", "Na", "K", "Rb", "Cs"]

    ion_dimer_equilibrium_distances = Dict(
        "LiF" => 1.5740842692,
        "NaF" => 1.9395644696,
        "KF" => 2.191804663,
        "RbF" => 2.3418174634,
        "CsF" => 2.3791859333999996,
        "LiCl" => 2.0273134221999998,
        "NaCl" => 2.3681930533999997,
        "KCl" => 2.6883189056,
        "RbCl" => 2.848191689,
        "CsCl" => 2.9408768782,
        "LiBr" => 2.1804909851999996,
        "NaBr" => 2.5125576179999998,
        "KBr" => 2.8474846406000003,
        "RbBr" => 3.0101616448,
        "CsBr" => 3.1139594843999996,
        "LiI" => 2.3928627104,
        "NaI" => 2.7130080771999996,
        "KI" => 3.0659928548,
        "RbI" => 3.2302740466,
        "CsI" => 3.3471488736,
    )

    for i_anion in eachindex(anion_labels)
        for i_cation in eachindex(cation_labels)
            geom = zeros(3, 3)
            geom[1] = ion_dimer_equilibrium_distances[string(cation_labels[i_cation], anion_labels[i_anion])]
            geom[9] = ion_dimer_equilibrium_distances[string(cation_labels[i_cation], anion_labels[i_anion])]
            labels = [anion_labels[i_anion], cation_labels[i_cation], anion_labels[i_anion]]
            scan_geoms = distance_scan(geom, (3, 2), -0.7, 6.0, 67, only_move_atom_1=true)
            mkpath(string(lowercase(anion_labels[i_anion]), "2_", lowercase(cation_labels[i_cation])))
            cd(string(lowercase(anion_labels[i_anion]), "2_", lowercase(cation_labels[i_cation])))
            write_xyz(
                string(lowercase(anion_labels[i_anion]), "2_", lowercase(cation_labels[i_cation]), "_mbe_eda_scan_geoms.xyz"),
                [labels for i in eachindex(scan_geoms)], scan_geoms
            )
            for i in eachindex(scan_geoms)
                write_mbe_inputs(
                    string(lowercase(anion_labels[i_anion]), "2_", lowercase(cation_labels[i_cation]), "_wb97xv_qzvppd_mbe_eda_scan_", i, ".in"),
                    scan_geoms[i],
                    labels, eda_input(), [[1], [2], [3]], 2
                )
            end
            cd("..")
        end
    end
end

function write_two_cation_one_anion_scans()
    anion_labels=["F", "Cl", "Br", "I"]
    cation_labels=["Li", "Na", "K", "Rb", "Cs"]

    ion_dimer_equilibrium_distances = Dict(
        "LiF" => 1.5740842692,
        "NaF" => 1.9395644696,
        "KF" => 2.191804663,
        "RbF" => 2.3418174634,
        "CsF" => 2.3791859333999996,
        "LiCl" => 2.0273134221999998,
        "NaCl" => 2.3681930533999997,
        "KCl" => 2.6883189056,
        "RbCl" => 2.848191689,
        "CsCl" => 2.9408768782,
        "LiBr" => 2.1804909851999996,
        "NaBr" => 2.5125576179999998,
        "KBr" => 2.8474846406000003,
        "RbBr" => 3.0101616448,
        "CsBr" => 3.1139594843999996,
        "LiI" => 2.3928627104,
        "NaI" => 2.7130080771999996,
        "KI" => 3.0659928548,
        "RbI" => 3.2302740466,
        "CsI" => 3.3471488736,
    )

    for i_anion in eachindex(anion_labels)
        for i_cation in eachindex(cation_labels)
            geom = zeros(3, 3)
            geom[1] = ion_dimer_equilibrium_distances[string(cation_labels[i_cation], anion_labels[i_anion])]
            geom[9] = ion_dimer_equilibrium_distances[string(cation_labels[i_cation], anion_labels[i_anion])]
            labels = [cation_labels[i_cation], anion_labels[i_anion], cation_labels[i_cation]]
            scan_geoms = distance_scan(geom, (3, 2), -0.7, 6.0, 67, only_move_atom_1=true)
            mkpath(string(lowercase(cation_labels[i_cation]), "2_", lowercase(anion_labels[i_anion])))
            cd(string(lowercase(cation_labels[i_cation]), "2_", lowercase(anion_labels[i_anion])))
            write_xyz(
                string(lowercase(cation_labels[i_cation]), "2_", lowercase(anion_labels[i_anion]), "_mbe_eda_scan_geoms.xyz"),
                [labels for i in eachindex(scan_geoms)], scan_geoms
            )
            for i in eachindex(scan_geoms)
                write_mbe_inputs(
                    string(lowercase(cation_labels[i_cation]), "2_", lowercase(anion_labels[i_anion]), "_wb97xv_qzvppd_mbe_eda_scan_", i, ".in"),
                    scan_geoms[i],
                    labels, eda_input(), [[1], [2], [3]], 2
                )
            end
            cd("..")
        end
    end
end

function write_water_two_cation_scans()
    cation_labels=["Li", "Na", "K", "Rb", "Cs"]

    ion_water_dimer_equilibrium_distances = Dict(
        "Li" => 1.8605727840215947,
        "Na" => 2.2478974336840563,
        "K" => 2.6463373359340663,
        "Rb" => 2.8333436608985725,
        "Cs" => 3.006992354480198,
    )

    for i_cation in eachindex(cation_labels)
        geom = zeros(3, 5)
        @views geom[:, 1] = [0.0, 0.11674231,  0.0]
        @views geom[:, 2] = [-0.76124067, -0.46696949,  0.0]
        @views geom[:, 3] = [ 0.76124067, -0.46696949,  0.0]
        θ = 52.25 * π / 180.0
        R_eq = ion_water_dimer_equilibrium_distances[cation_labels[i_cation]]
        @views geom[:, 4] = [ R_eq * sin(θ), 0.11674231 + R_eq * cos(θ),  0.0]
        @views geom[:, 5] = [-R_eq * sin(θ), 0.11674231 + R_eq * cos(θ),  0.0]
        labels = ["O", "H", "H", cation_labels[i_cation], cation_labels[i_cation]]

        mkpath(string("w1_", lowercase(cation_labels[i_cation]), "2"))
        cd(string("w1_", lowercase(cation_labels[i_cation]), "2"))
        O_ion_distances = [(R_eq-0.7):0.1:(R_eq+6.0)...]
        O_ion_1_vec = normalize(geom[:, 4] - geom[:, 1])
        O_ion_2_vec = normalize(geom[:, 5] - geom[:, 1])
        ion_1_positions = [geom[:, 1] + O_ion_distances[i] * O_ion_1_vec for i in eachindex(O_ion_distances)]
        ion_2_positions = [geom[:, 1] + O_ion_distances[i] * O_ion_2_vec for i in eachindex(O_ion_distances)]
        scan_geoms = [copy(geom) for _ in eachindex(O_ion_distances)]
        for i in eachindex(scan_geoms)
            @views scan_geoms[i][:, 4] = ion_1_positions[i]
            @views scan_geoms[i][:, 5] = ion_2_positions[i]
        end
        write_xyz(
            string(string("w1_", lowercase(cation_labels[i_cation]), "2_mbe_eda_scan_geoms.xyz")),
            [labels for i in eachindex(scan_geoms)], scan_geoms
        )
        for i in eachindex(scan_geoms)
            write_mbe_inputs(
                string("w1_", lowercase(cation_labels[i_cation]), "_wb97xv_qzvppd_mbe_eda_scan_", i, ".in"),
                scan_geoms[i],
                labels, eda_input(), [[1,2,3], [4], [5]], 2
            )
        end
        cd("..")
    end
end

function write_water_two_anion_scans()
    anion_labels=["F", "Cl", "Br", "I"]

    ion_water_dimer_equilibrium_distances = Dict(
        "F" => 2.4698184455120584,
        "Cl" => 3.1429790118854157,
        "Br" => 3.340785263204781,
        "I" => 3.596534030535338,
    )

    for i_anion in eachindex(anion_labels)
        geom = zeros(3, 5)
        @views geom[:, 1] = [0.0, 0.11674231,  0.0]
        @views geom[:, 2] = [-0.76124067, -0.46696949,  0.0]
        @views geom[:, 3] = [ 0.76124067, -0.46696949,  0.0]
        θ = 52.25 * π / 180.0
        R_eq = ion_water_dimer_equilibrium_distances[anion_labels[i_anion]]
        @views geom[:, 4] = [ R_eq * sin(θ), -(R_eq * cos(θ) - 0.11674231),  0.0]
        @views geom[:, 5] = [-R_eq * sin(θ), -(R_eq * cos(θ) - 0.11674231),  0.0]
        labels = ["O", "H", "H", anion_labels[i_anion], anion_labels[i_anion]]

        mkpath(string("w1_", lowercase(anion_labels[i_anion]), "2"))
        cd(string("w1_", lowercase(anion_labels[i_anion]), "2"))
        O_ion_distances = [(R_eq-0.7):0.1:(R_eq+6.0)...]
        O_ion_1_vec = normalize(geom[:, 4] - geom[:, 1])
        O_ion_2_vec = normalize(geom[:, 5] - geom[:, 1])
        ion_1_positions = [geom[:, 1] + O_ion_distances[i] * O_ion_1_vec for i in eachindex(O_ion_distances)]
        ion_2_positions = [geom[:, 1] + O_ion_distances[i] * O_ion_2_vec for i in eachindex(O_ion_distances)]
        scan_geoms = [copy(geom) for _ in eachindex(O_ion_distances)]
        for i in eachindex(scan_geoms)
            @views scan_geoms[i][:, 4] = ion_1_positions[i]
            @views scan_geoms[i][:, 5] = ion_2_positions[i]
        end
        write_xyz(
            string(string("w1_", lowercase(anion_labels[i_anion]), "2_mbe_eda_scan_geoms.xyz")),
            [labels for i in eachindex(scan_geoms)], scan_geoms
        )
        for i in eachindex(scan_geoms)
            write_mbe_inputs(
                string("w1_", lowercase(anion_labels[i_anion]), "_wb97xv_qzvppd_mbe_eda_scan_", i, ".in"),
                scan_geoms[i],
                labels, eda_input(), [[1,2,3], [4], [5]], 2
            )
        end
        cd("..")
    end
end

function write_two_water_cation_scans()
    cation_labels=["Li", "Na", "K", "Rb", "Cs"]

    ion_w2_geoms = Dict(
        "Li" => reduce(hcat, 
            [[1.0086122651,    -2.3304560468,    -0.6411195767],
            [0.2833749318,    -2.4439345995,    -1.2645543490],
            [1.3311380979,    -3.2178221941,    -0.4513522752],
            [2.2836905487,     0.9025323664,     0.7196429755],
            [3.0864387321,     1.3712590588,     0.4681466405],
            [1.8686491843,     1.4413698461,     1.4013008997],
            [1.6631197179,    -0.7234972398,     0.0441441063]]
        ),
        "Na" => reduce(hcat, 
            [[0.8761399806,    -2.6538877812,    -0.7954545182],
            [0.1328809182,    -2.7764971875,    -1.3938432440],
            [1.2132622896,    -3.5399437753,    -0.6318846411],
            [2.4178808133,     1.2250842058,     0.8734640167],
            [3.2354362500,     1.6798238980,     0.6490683233],
            [1.9958573697,     1.7829873638,     1.5339585443],
            [1.6535658564,    -0.7181155326,     0.0408999401]]
        ),
        "K" => reduce(hcat, 
            [[0.8418321960,    -2.7697259965,    -0.8671209369],
            [0.2692075317,    -3.2830826000,    -1.4448755080],
            [1.5043661346,    -3.3975359195,    -0.5639041587],
            [2.4809799970,     1.3223039645,     0.9537398503],
            [3.4199515682,     1.3812022809,     0.7540431371],
            [2.3478618203,     1.9435163827,     1.6758708740],
            [0.6608242300,    -0.1972269212,    -0.2315448365]]
        ),
        "Rb" => reduce(hcat, 
            [[0.8021749789,    -2.9019296423,    -0.9120030164],
            [0.3292677748,    -3.3955504511,    -1.5883722600],
            [1.3740132038,    -3.5529261793,    -0.4949404971],
            [2.5457379538,     1.4428068821,     1.0056380062],
            [3.4475427955,     1.6013413268,     0.7117493872],
            [2.4639907118,     1.9535152240,     1.8163710596],
            [0.5622960591,    -0.1478059693,    -0.2622342584]]
        ),
        "Cs" => reduce(hcat, 
            [[0.8361879475,    -2.8727067367,    -0.8907335493],
            [0.4488805673,    -3.4506121372,    -1.5546185096],
            [1.4558906036,    -3.4335755195,    -0.4153097435],
            [2.5515292226,     1.3905549775,     0.9959254479],
            [3.4592625910,     1.4549660282,     0.6852190611],
            [2.5462566688,     1.8767008181,     1.8256097414],
            [0.2270158769,     0.0341237606,    -0.3698840267]]
        )
    )

    for i_cation in eachindex(cation_labels)
        geom = ion_w2_geoms[cation_labels[i_cation]]
        labels = ["O", "H", "H", "O", "H", "H", cation_labels[i_cation]]

        mkpath(string("w2_", lowercase(cation_labels[i_cation])))
        cd(string("w2_", lowercase(cation_labels[i_cation])))
        scan_geoms_1 = distance_scan(geom, ((1, 7), ([3, 2], Int[])), -0.7, 6.0, 68, true)
        scan_geoms_2 = distance_scan(geom, ((4, 7), ([6, 5], Int[])), -0.7, 6.0, 68, true)
        scan_geoms = [copy(geom) for _ in eachindex(scan_geoms_1)]
        for i in eachindex(scan_geoms)
            @views scan_geoms[i][:, 1:3] = scan_geoms_1[i][:, 1:3]
            @views scan_geoms[i][:, 4:6] = scan_geoms_2[i][:, 4:6]
        end
        write_xyz(
            string(string("w2_", lowercase(cation_labels[i_cation]), "_mbe_eda_scan_geoms.xyz")),
            [labels for i in eachindex(scan_geoms)], scan_geoms
        )
        for i in eachindex(scan_geoms)
            write_mbe_inputs(
                string("w2_", lowercase(cation_labels[i_cation]), "_wb97xv_qzvppd_mbe_eda_scan_", i, ".in"),
                scan_geoms[i],
                labels, eda_input(), [[1,2,3], [4, 5, 6], [7]], 2
            )
        end
        cd("..")
    end
end

function write_two_water_anion_scans()
    anion_labels=["F", "Cl", "Br", "I"]

    ion_w2_geoms = Dict(
        "F" => reduce(hcat, 
            [[ 1.6824375292,    -1.7459878131,    -0.6328222236],
            [ 1.5162032341,    -2.5765979062,    -0.1862421495],
            [ 1.1741033556,    -1.0785875993,    -0.0577732691],
            [-1.9173351853,    -1.1533975619,     0.1591721904],
            [-1.0551231640,    -0.7377854573,     0.4855534329],
            [-1.6420578148,    -1.5606133938,    -0.6634069398],
            [ 0.3375010451,    -0.2037652682,     0.8361069588]]
        ),
        "Cl" => reduce(hcat, 
            [[ 1.6869670072,    -1.8531641384,    -0.1198581996],
            [ 0.8102829717,    -2.2309983142,    -0.2601151959],
            [ 1.4614727287,    -0.9103831098,    -0.0317187892],
            [-1.3251718618,    -1.9272350901,    -0.1936114239],
            [-1.0094298398,    -0.9936364518,    -0.2540530639],
            [-1.4697515597,    -2.0263648271,     0.7494118297],
            [-0.0586404462,     0.8850469314,     0.0505328427]]
        ),
        "Br" => reduce(hcat, 
            [[ 1.6991952875,    -1.8993399422,    -0.1274513446],
            [ 0.8056381531,    -2.2448246646,    -0.2440321618],
            [ 1.5190135881,    -0.9501116683,    -0.0360146043],
            [-1.3194918069,    -1.9611918705,    -0.1942496295],
            [-1.0584412739,    -1.0177736371,    -0.2807419311],
            [-1.4615528975,    -2.0407424524,     0.7513402293],
            [-0.0886320504,     1.0572492351,     0.0717374421]]
        ),
        "I" => reduce(hcat, 
            [[ 1.6986473327,    -1.9639029435,    -0.1312274604],
            [ 0.7823115723,    -2.2536683162,    -0.2279229839],
            [ 1.5893933601,    -1.0070597582,    -0.0363129265],
            [-1.3037920656,    -2.0122930500,    -0.1914678937],
            [-1.1100999228,    -1.0650960944,    -0.3370086932],
            [-1.4197728402,    -2.0447934301,     0.7610489098],
            [-0.1409584365,     1.2900785924,     0.1034790478]]
        )
    )

    for i_anion in eachindex(anion_labels)
        geom = ion_w2_geoms[anion_labels[i_anion]]
        labels = ["O", "H", "H", "O", "H", "H", anion_labels[i_anion]]

        mkpath(string("w2_", lowercase(anion_labels[i_anion])))
        cd(string("w2_", lowercase(anion_labels[i_anion])))
        com_vec = reduce(vcat, center_of_mass(geom, labels))
        scan_geoms = distance_scan_from_point(geom, 7, com_vec, -0.7, 6.0, 68)
        write_xyz(
            string(string("w2_", lowercase(anion_labels[i_anion]), "_mbe_eda_scan_geoms.xyz")),
            [labels for i in eachindex(scan_geoms)], scan_geoms
        )
        for i in eachindex(scan_geoms)
            write_mbe_inputs(
                string("w2_", lowercase(anion_labels[i_anion]), "_wb97xv_qzvppd_mbe_eda_scan_", i, ".in"),
                scan_geoms[i],
                labels, eda_input(), [[1,2,3], [4, 5, 6], [7]], 2
            )
        end
        cd("..")
    end
end

"""
Generates copies of a coordinate matrix with appropriate displaced geometries
for doing a finite difference force calculation.
"""
function get_finite_difference_structures(coords::Matrix{Float64}, step_size=1e-3)
    fd_coords = [copy(coords) for _ in 1:(2*length(coords))]
    dof_index = 1
    for i in 1:2:length(fd_coords)
        fd_coords[i][dof_index] += step_size
        fd_coords[i+1][dof_index] -= step_size
        dof_index += 1
    end
    return fd_coords
end

function write_electrostatic_potential_input(infile_name::String, coords::Vector{MVector{3, Float64}}, labels::Vector{String}, charge::Int, multplicity::Int, min_factor_of_vdw_radius::Float64=0.9, max_factor_of_vdw_radius::Float64=1.5)
    esp_grid = generate_electrostatic_potential_grid(coords, labels, min_factor_of_vdw_radius, max_factor_of_vdw_radius)
    esp_grid /= 0.529177

    num_grid_points = length(esp_grid)

    writedlm("ESPGrid", esp_grid)

    rem_input = "\$rem
  mem_total  16000
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
  ESP_GRID     +$num_grid_points
\$end

\$plots
  comment line
  1   0.0   0.0
  1   0.0   0.0
  1   0.0   0.0
  0  0  0  0
  0
\$end
"

write_input_file(infile_name, reduce(hcat, coords), labels, rem_input, charge, multplicity)
    
end

function generate_electrostatic_potential_grid(coords::Vector{MVector{3, Float64}}, labels::Vector{String}, min_factor_of_vdw_radius::Float64=0.9, max_factor_of_vdw_radius::Float64=1.5)
    com = center_of_mass(reduce(hcat, coords), labels)
    vdw_radii = [vdw_radius(label) for label in labels]
    
    gridx = (com[1] - 6.0):0.2:(com[1] + 6.0)
    gridy = (com[2] - 6.0):0.2:(com[2] + 6.0)
    gridz = (com[3] - 6.0):0.2:(com[3] + 6.0)

    grid_points = MVector{3, Float64}[]
    for i in eachindex(gridx)
        for j in eachindex(gridy)
            for k in eachindex(gridz)
                point = MVector{3, Float64}(gridx[i], gridy[j], gridz[k])
                push!(grid_points, point)
            end
        end
    end

    # @ROBUSTNESS: This approach won't work for large molecules
    # since a grid point could be accepted by one atom even though it
    # is sitting on top of another atom. Could just set a minimum distance?
    accepted_indices = Int[]
    for i_point in eachindex(grid_points)
        is_accepted_by_any_atom = false
        for i in eachindex(coords)
            # Find points to accept for this atom
            dist_to_point = norm(grid_points[i_point] - coords[i])
            if ((dist_to_point > min_factor_of_vdw_radius * vdw_radii[i]) & (dist_to_point < max_factor_of_vdw_radius * vdw_radii[i]))
                is_accepted_by_any_atom = true
                break
            end
        end
        if is_accepted_by_any_atom
            push!(accepted_indices, i_point)
        end
    end

    return grid_points[accepted_indices]
end

function perlmutter_slurm_script_string(infile_prefix::AbstractString, batch_number::Int, index_start::Int, num_per_batch::Int)
    index_end = index_start + num_per_batch - 1
    full_file = string(infile_prefix, "_batch_", batch_number)
    return string("#!/bin/bash
#SBATCH -A m2834
#SBATCH -t 24:00:00
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
#SBATCH -A m2834
#SBATCH -t 24:00:00
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