
mutable struct MonteCarlo
    kT::Float64
    current_energy::Float64
    
    MonteCarlo(temperature::Float64, current_energy::Float64) = new(temperature * 3.166811563*10^(-6), current_energy)
end

MonteCarlo(temperature::Float64) = MonteCarlo(temperature, 0.0)

"""
Evaluates metropolis criteria given 
"""
function evaluate_metropolis_criteria(energy::Float64, mc::MonteCarlo)
    # HERE!!!
end