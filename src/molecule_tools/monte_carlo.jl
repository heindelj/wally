using Distributions

mutable struct MonteCarlo
    kT::Float64
    current_energy::Float64
    
    MonteCarlo(temperature::Float64, current_energy::Float64) = new(temperature * 3.166811563*10^(-6), current_energy)
end

MonteCarlo(temperature::Float64) = MonteCarlo(temperature, maxintfloat(Float64))

"""
Evaluates metropolis criteria given a proposed energy. If the move is accepted
then we return true otherwise false.
"""
function evaluate_metropolis_criteria(energy::Float64, mc::MonteCarlo)
    if energy < mc.current_energy
        return true
    end
    p = rand(Uniform(0, 1))
    if p < exp(-(energy - mc.current_energy) / mc.kT)
        return true
    end
    return false
end