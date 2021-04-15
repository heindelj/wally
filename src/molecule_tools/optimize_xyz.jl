include("read_xyz.jl")
include("call_potential.jl")
using Optim

function optimize_xyz(geom::AbstractVecOrMat{<:AbstractFloat}, potential::AbstractPotential; g_tol=0.0001, iterations=2500, show_every=25, show_trace::Bool=true, kwargs...)
    shape = size(geom)
    results = optimize(geom -> get_energy(potential, geom; kwargs...),
                    (grads, x) -> get_gradients!(potential, grads, x; kwargs...),
                           geom,
                           LBFGS(linesearch=Optim.LineSearches.MoreThuente()),
                           Optim.Options(g_tol=g_tol,
                                         show_trace=show_trace,
                                         show_every=show_every,
                                         iterations=iterations))
    final_geom = reshape(Optim.minimizer(results), shape)
    return (Optim.minimum(results), final_geom)
end

function optimize_structures(structures::AbstractArray{Array{Float64,2}, 1}, potential::AbstractPotential; show_trace::Bool=false, copy_construct_potential=false, kwargs...)
    """
    Optimizes many guess structures suitable for the provided potential.
    We divide the work among as many processes as possible.
    copy_construct_potential provides the option to copy construct the AbstractPotential as typeof(potential)(potential).
    This allows the use of potentials which are not thread safe, as is often the case with potentials written in fortran by scientists.
    """
    #future_results = Array{Future, 1}(undef, length(structures))
    #@sync for i in 1:length(structures)
    #    if copy_construct_potential
    #        future_results[i] = @spawnat :any optimize_xyz(structures[i], typeof(potential)(potential), show_trace=show_trace; kwargs...)
    #    else
    #        future_results[i] = @spawnat :any optimize_xyz(structures[i], potential, show_trace=show_trace; kwargs...)
    #    end
    #end

    # TEST THIS!!!!!
    if copy_construct_potential
        results = pmap(x -> optimize_xyz(x, typeof(potential)(potential), show_trace=show_trace; kwargs...), structures)
    else
        results = pmap(x -> optimize_xyz(x, potential, show_trace=show_trace; kwargs...), structures)
    end

    energies, geoms = collect(zip(fetch.(results)...))
    return [energies...], [geoms...]
end