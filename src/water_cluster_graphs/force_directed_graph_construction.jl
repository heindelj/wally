using Graphs, Optim
include("tcode_to_cluster_structure.jl")

struct LayoutForces
    attractive_energy::Function
    attractive_force::Function
    repulsive_energy::Union{Function, Nothing}
    repulsive_force::Union{Function, Nothing}
end

#LayoutForces(attractive_force_name::Symbol) = LayoutForces(attractive_force::Function, nothing)

function harmonic_energy(coords::Matrix{T}, g::Union{SimpleGraph, SimpleDiGraph}, force_constant::T=1.0, optimal_distance::T=1.0) where T <: AbstractFloat
    energy::T = 0.0
    for i in vertices(g)
        for j in all_neighbors(g, i)
            @views energy += 0.5 * force_constant * (norm(coords[:,i] - coords[:, j]) - optimal_distance)^2
        end
    end
    return 0.5 * energy # I'm double-counting right now...
end

function harmonic_gradient!(grads::Matrix{T}, coords::Matrix{T}, g::Union{SimpleGraph, SimpleDiGraph}, force_constant::T=1.0, optimal_distance::T=1.0) where T <: AbstractFloat
    for i in vertices(g)
        for j in all_neighbors(g, i)
            @views grads[:,i] += (coords[:,i] - coords[:,j]) / norm(coords[:,i] - coords[:,j]) * force_constant * (norm(coords[:,i] - coords[:, j]) - optimal_distance) 
        end
    end
end

function harmonic_gradient(coords::Matrix{T}, g::Union{SimpleGraph, SimpleDiGraph}, force_constant::T=1.0, optimal_distance::T=1.0) where T <: AbstractFloat
    grads = zero(coords)
    for i in vertices(g)
        for j in all_neighbors(g, i)
            @views grads[:,i] += (coords[:,i] - coords[:,j]) / norm(coords[:,i] - coords[:,j]) * force_constant * (norm(coords[:,i] - coords[:, j]) - optimal_distance) 
        end
    end
    return grads
end

function coulomb_repulsion_energy(coords::Matrix{T}, g::Union{SimpleGraph, SimpleDiGraph}, C::T=0.01) where T <: AbstractFloat
    energy::T = 0.0
    for i in vertices(g)
        for j in setdiff(1:length(g.fadjlist), all_neighbors(g, i), i) # loop over non-neighbors
            @views energy += C / norm(coords[:,i] - coords[:, j])
        end
    end
    return energy
end

function coulomb_repulsion_gradient(coords::Matrix{T}, g::Union{SimpleGraph, SimpleDiGraph}, C::T=0.01) where T <: AbstractFloat
    grads = zero(coords)
    for i in vertices(g)
        for j in setdiff(1:length(g.fadjlist), all_neighbors(g, i), i) # loop over non-neighbors
            @views grads[:,i] -= C * (coords[:,i] - coords[:,j]) / norm(coords[:,i] - coords[:, j])^2
        end
    end
    return grads
end

function total_energy(coords::Matrix{T}, g::Union{SimpleGraph, SimpleDiGraph}, force_constant::T=1.0, C::T=0.01, optimal_distance::T=2.8) where T <: AbstractFloat
    return harmonic_energy(coords, g, force_constant, optimal_distance) + coulomb_repulsion_energy(coords, g, C)
end

function total_gradient(coords::Matrix{T}, g::Union{SimpleGraph, SimpleDiGraph}, force_constant::T=1.0, C::T=0.01, optimal_distance::T=2.8) where T <: AbstractFloat
    return harmonic_gradient(coords, g, force_constant, optimal_distance) + coulomb_repulsion_gradient(coords, g, C)
end

function guess_cluster_structure_from_graph(g::SimpleDiGraph, guess_structure::Union{Matrix{Float64}, Nothing}=nothing, force_constant::Float64=1.0, C::Float64=0.01, optimal_distance::Float64=2.78)
    if guess_structure === nothing 
        guess_structure = randn(3, length(g.fadjlist)) # Try placing the coordinates more thoughtfully. Perhaps a sphere would be better?
    end
    res = optimize(x -> total_energy(x, g, force_constant, C, optimal_distance), x -> total_gradient(x, g, force_constant, C, optimal_distance), guess_structure, LBFGS(), Optim.Options(iterations=5000, g_tol=1e-6); inplace=false)
    if !res.g_converged
        println("Warning: Failed to optimize the graph coordinates using a spring-charge model. Proceeding with the coordinates from the last step.")
    end
    # get the optimized oxygen framework coordinates
    guess_structure = Optim.minimizer(res)
    
    # place the oxygen only coordinates into a matrix of future OHH coordinates
    final_coords = zeros(3, 3*length(g.fadjlist))
    final_coords[:,1:3:end] = guess_structure[:,:]
    place_hydrogen_atoms_in_oxygen_framework!(tcode_from_digraph(g), final_coords)
    return final_coords, res.g_converged
end
