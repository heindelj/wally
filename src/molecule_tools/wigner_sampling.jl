using ClassicalOrthogonalPolynomials

include("harmonic_frequencies.jl")

function sample_harmonic_wigner_density(ω::Vector{T}, μ::Vector{T}, temperature::Union{Int, T}, num_states::Int=10) where T <: Real
    # see: http://www.rsc.org/suppdata/c8/cp/c8cp03273d/c8cp03273d2.pdf
    probabilities_all_states::Vector{Vector{T}} = [zeros(num_states) for _ in 1:length(ω)]
    ω_all::Vector{Vector{T}} = [[ω[i] * n for n in LinRange(0.5, (num_states - 1) + 0.5, num_states)] for i in 1:length(ω)]
    for (k, probabilities_state_k) in enumerate(probabilities_all_states)
        Z_state_k = exp(-ω[k]/(2 * temperature * conversion(:kelvin, :au_temperature))) / (1 - exp(-ω[k]/(temperature * conversion(:kelvin, :au_temperature))))
        for n in 1:length(probabilities_state_k)
            probabilities_state_k[n] = exp(-ω_all[k][n] / (temperature * conversion(:kelvin, :au_temperature))) / Z_state_k
        end
    end
return probabilities_all_states
end