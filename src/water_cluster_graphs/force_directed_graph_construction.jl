using Graphs

struct LayoutForces
    attractive_energy::Function
    attractive_force::Function
    repulsive_energy::Union{Function, Nothing}
    repulsive_force::Union{Function, Nothing}
end

#LayoutForces(attractive_force_name::Symbol) = LayoutForces(attractive_force::Function, nothing)

function harmonic_energy(coords::Matrix{T}, g::Graph, force_constant::T=1.0, optimal_distance::T=1.0) where T <: AbstractFloat
    energy::T = 0.0
    for i in vertices(g)
        for j in neighbors(g, i)
            @views energy += 0.5 * force_constant * (norm(coords[:,i] - coords[:, j]) - optimal_distance)^2
        end
    end
    return 0.5 * energy # I'm double-dounting right now...
end

function harmonic_gradient!(grads::Matrix{T}, coords::Matrix{T}, g::Graph, force_constant::T=1.0, optimal_distance::T=1.0) where T <: AbstractFloat
    for i in vertices(g)
        for j in neighbors(g, i)
            @views grads[:,i] += (coords[:,i] - coords[:,j]) / norm(coords[:,i] - coords[:,j]) * force_constant * (norm(coords[:,i] - coords[:, j]) - optimal_distance) 
        end
    end
end

function harmonic_gradient(coords::Matrix{T}, g::Graph, force_constant::T=1.0, optimal_distance::T=1.0) where T <: AbstractFloat
    grads = zero(coords)
    for i in vertices(g)
        for j in neighbors(g, i)
            @views grads[:,i] += (coords[:,i] - coords[:,j]) / norm(coords[:,i] - coords[:,j]) * force_constant * (norm(coords[:,i] - coords[:, j]) - optimal_distance) 
        end
    end
    return grads
end

function coulomb_repulsion_energy(coords::Matrix{T}, g::Graph, C::T=0.01) where T <: AbstractFloat
    energy::T = 0.0
    for i in vertices(g)
        for j in setdiff(1:length(g.fadjlist), neighbors(g, i), i) # loop over non-neighbors
            @views energy += C / norm(coords[:,i] - coords[:, j])
        end
    end
    return energy
end

function coulomb_repulsion_gradient(coords::Matrix{T}, g::Graph, C::T=0.01) where T <: AbstractFloat
    grads = zero(coords)
    for i in vertices(g)
        for j in setdiff(1:length(g.fadjlist), neighbors(g, i), i) # loop over non-neighbors
            @views grads[:,i] -= C * (coords[:,i] - coords[:,j]) / norm(coords[:,i] - coords[:, j])^2
        end
    end
    return grads
end

function total_energy(coords::Matrix{T}, g::Graph, force_constant::T=1.0, C::T=0.01, optimal_distance::T=2.8) where T <: AbstractFloat
    return harmonic_energy(coords, g, force_constant, optimal_distance) + coulomb_repulsion_energy(coords, g, C)
end

function total_gradient(coords::Matrix{T}, g::Graph, force_constant::T=1.0, C::T=0.01, optimal_distance::T=2.8) where T <: AbstractFloat
    return harmonic_gradient(coords, g, force_constant, optimal_distance) + coulomb_repulsion_gradient(coords, g, C)
end
