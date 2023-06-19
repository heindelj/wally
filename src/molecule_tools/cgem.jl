using StaticArrays, LinearAlgebra, NearestNeighbors, Optim, SpecialFunctions
include("read_xyz.jl")

mutable struct CGeMResults
    energy::Float64
    grads::Union{Nothing, Vector{MVector{3, Float64}}}
end
CGeMResults() = CGeMResults(0.0, nothing)

mutable struct CGeM
    results::CGeMResults
    params::Dict{Symbol, Float64}
end

function get_cgem_parameters(parameters_name::Symbol)

    parameter_sets = Dict(
        :protein_old => Dict(
            :ω_c => 0.1521927977681796,
            :γ_s => 5.219843769327116,
            :λ_c => 2.1031572388898803,
            :λ_s => 2.1031572388898803,
            :r_shell => 0.7078603566311076,
            :shell_ip => 19.95601666013147,
            :r_atomic_1 => 0.691693289916444,
            :r_atomic_6 => 0.5972202599148441,
            :r_atomic_7 => 0.4334608877498918,
            :r_atomic_8 => 0.29962573169522294,
            :r_atomic_16 => 0.5782979621043117,
            :r_atomic_17 => 0.2810762136767955,
            :r_atomic_6001 => 0.5854636993144134,
            :r_atomic_7001 => 0.841073011324028,
            :r_atomic_110 => 0.787437880615057,
            :r_atomic_610 => 0.5900136261747168,
            :r_atomic_820 => 0.593152711765547,
            :atomic_ip_1 => -16.05545878746296,
            :atomic_ip_6 => -18.500350837982577,
            :atomic_ip_7 => -21.736295347371925,
            :atomic_ip_8 => -23.58553472898272,
            :atomic_ip_16 => -21.32897255399771,
            :atomic_ip_17 => -25.3175033857104,
            :atomic_ip_6001 => -31.26050507889707,
            :atomic_ip_7001 => -27.338919928283715,
            :atomic_ip_110 => -15.425280590955042,
            :atomic_ip_610 => -19.483006174304194,
            :atomic_ip_820 => -22.09424547196936,
        ),
        :protein => Dict(
            :ω_c => 0.3235448156735446,
            :γ_s => 1.4261506814074827,
            :λ_c => 2.509042103587565,
            :λ_s => 2.509042103587565,
            :r_shell => 0.7630885036587745,
            :shell_ip => 18.491089435641626,
            :r_atomic_1 => 0.7603626359305602,
            :r_atomic_6 => 0.6490919588913426,
            :r_atomic_7 => 0.42274440135326463,
            :r_atomic_8 => 0.4566521878952215,
            :r_atomic_16 => 0.7512024171695153,
            :r_atomic_17 => 0.3690567918027655,
            :r_atomic_6001 => 0.27230905933249644,
            :r_atomic_7001 => 0.9515021769620915,
            :r_atomic_110 => 0.7473519365471248,
            :r_atomic_610 => 0.6798051821334476,
            :r_atomic_820 => 0.46040929817223414,
            :atomic_ip_1 => -16.97764304538409,
            :atomic_ip_6 => -19.280819173565664,
            :atomic_ip_7 => -20.61719083753313,
            :atomic_ip_8 => -24.14970943441986,
            :atomic_ip_16 => -21.992728316766357,
            :atomic_ip_17 => -26.100978109039858,
            :atomic_ip_6001 => -31.983843957258344,
            :atomic_ip_7001 => -35.15265056746553,
            :atomic_ip_110 => -17.03928126124472,
            :atomic_ip_610 => -24.33249616950374,
            :atomic_ip_820 => -24.51370782536112
        ),
    )
    if haskey(parameter_sets, parameters_name)
        return parameter_sets[parameters_name]
    end
    @assert false "Don't have parameter set by the name $parameters_name."
end

function build_cgem_model(parameters_name::Symbol)
    return CGeM(CGeMResults(), get_cgem_parameters(parameters_name))
end

function labels_to_nums(labels::Vector{String})
    label_to_num = Dict(
        "H" => 1,
        "O" => 8
    )
    nums = zeros(Int, length(labels))
    for i in eachindex(labels)
        nums[i] = label_to_num[labels[i]]
    end
    return nums
end

function get_α_cores_and_shells(atomic_nums::Vector{Int}, params::Dict{Symbol, Float64})
    α_cores = zeros(length(atomic_nums))
    for i in eachindex(α_cores)
        α_cores[i] = params[:λ_c] / (2 * params[Symbol(:r_atomic_, atomic_nums[i])]^2)
    end
    α_shell = params[:λ_s] / (2 * params[:r_shell]^2)
    return α_shell, α_cores
end

# TARGETS #
# coulomb = -450554.71533867717
# gauss   = 13219.456515468784
# ^^^ Reference coordinates. All positions the same.


"""
Takes an array of coords for both cores and shells. Computes the
core-core, core-shell, and shell-shell interactions.
"""
function pairwise_coulomb_energy(
    coords_cores::Vector{MVector{3, Float64}},
    coords_shells::Vector{MVector{3, Float64}},
    atomic_nums::Vector{Int},
    params::Dict{Symbol, Float64},
    q_core::Float64=1.0,
    q_shell::Float64=-1.0,
    r_cut::Float64=20.0,
    zero_threshold::Float64=10^-8
)
    α_shell, α_cores = get_α_cores_and_shells(atomic_nums, params)

    nl_cores  = KDTree(coords_cores, reorder=false)
    nl_shells = KDTree(coords_shells, reorder=false)
    energy = 0.0
    # Core-Core interactions
    for i in eachindex(nl_cores.data)
        for j in inrange(nl, nl_cores.data[i], r_cut)
            if i < j
                r_ij = norm(nl_cores.data[i] - nl_cores.data[j])
                if r_ij < zero_threshold
                    energy += 2 * q_core * q_core / π * sqrt(α_cores[i] * α_cores[j] / (α_cores[i] + α_cores[j]))
                else
                    energy += q_core * q_core / r_ij * erf(sqrt(α_cores[i] * α_cores[j] / (α_cores[i] + α_cores[j])) * r_ij)
                end
            end
        end
    end
    return energy
end
