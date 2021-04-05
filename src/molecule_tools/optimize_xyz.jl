include("call_potential.jl")
include("read_xyz.jl")
using Optim

function optimize_xyz(geom::AbstractVecOrMat{Float64}, potential::AbstractPotential; g_tol=0.0001, iterations=2000, show_every=10, show_trace::Bool=true)
    shape = size(geom)
    results = optimize(geom -> get_energy(potential, vec(geom), reshape_coords=true),
                    (grads, x) -> get_gradients!(potential, grads, x, reshape_coords=true),
                           geom,
                           LBFGS(),
                           Optim.Options(g_tol=g_tol,
                                         show_trace=show_trace,
                                         show_every=show_every,
                                         iterations=iterations))
    final_geom = reshape(Optim.minimizer(results), shape)
    return (Optim.minimum(results), final_geom)
end

function optimize_structures(structures::AbstractArray{Array{Float64,2}, 1}, potential::AbstractPotential; show_trace::Bool=false)
    """
    Optimizes many guess structures suitable for the provided potential.
    """
    energies = zeros(length(structures))
    final_structures = copy(structures)
    for i in 1:length(structures)
        energies[i], final_structures[i] = optimize_xyz(structures[i], potential, show_trace=show_trace)
    end
    return energies, final_structures
end
