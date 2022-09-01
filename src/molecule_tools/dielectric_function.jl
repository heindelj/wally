include("vdw_radii.jl")
include("molecular_axes.jl")
using StaticArrays, SpecialFunctions, LinearAlgebra, ProgressBars, GLMakie

# This file contains functions needed to discretize the dielectric function
# for doing cluster-continuum calculations. Namely, we first determine the
# solute cavity, which is roughly the SAS of the cluster, while ensuring
# the dielectric function inside the molecular cavity is ϵ=1.
# Then, we interpolate between the cavity and solvent dielectrics using
# a product of atom-centered switching functions described in:
# J. Chem. Phys. 148, 222834 (2018); https://doi.org/10.1063/1.5023916
# Finally, we also need to interpolate the dielectric between the cavity
# and a molecular interface (e.g. the air-water interface).
# This is done using the usual tanh function fit to the Gibbs dividing surface.
#
# All utilized vdw radii come from the bondi set plus additional values from:
# J Phys Chem A. 2009 May 14; 113(19): 5806–5812.
# Notably, we use the H vdw radius of 1.1 which was recommended after Bondi's
# original data which suggested a value of 1.2.

mutable struct DielectricFunction
    grid_x::LinRange{Float64, Int64}
    grid_y::LinRange{Float64, Int64}
    grid_z::LinRange{Float64, Int64}
    ϵ::Array{Float64, 3}
end

DielectricFunction(grid_x::AbstractRange, grid_y::AbstractRange, grid_z::AbstractRange) =
DielectricFunction(LinRange(grid_x), LinRange(grid_y), LinRange(grid_z),
                   zeros(length(grid_x), length(grid_y), length(grid_z)))

DielectricFunction(grid::AbstractRange) =
DielectricFunction(LinRange(grid), LinRange(grid), LinRange(grid),
                   zeros(length(grid), length(grid), length(grid)))

function S_α(r_grid::SVector{3, Float64}, R_α::SVector{3, Float64}, r_vdw::Float64, vdw_scale::Float64=1.2, Δ::Float64=0.265)
    """
    Calculates the switching function used in 
    J. Chem. Phys. 148, 222834 (2018); https://doi.org/10.1063/1.5023916
    r_grid is the grid point being evaluated, R_α is the atomic center,
    r_vdw is the vdw radius of that atom, vdw_scale is the factor by which
    r_vdw will be multiplied, and Δ sets the scale of the switching function.
    The latter two have recommended values which we set by default.
    """
    return 0.5 * (1.0 + erf((norm(r_grid - R_α) - vdw_scale * r_vdw) / Δ))
end

function S_α_mSAS(r_grid::SVector{3, Float64}, R_α::SVector{3, Float64}, r_vdw::Float64, r_probe::Float64=1.2, Δ::Float64=0.265)
    """
    Calculates the switching function used in 
    J. Chem. Phys. 148, 222834 (2018); https://doi.org/10.1063/1.5023916
    This is a modified solvent-accessible-surface which is the vdw surface
    with an extra distance of 0.7 ansgtroms added.
    """
    return 0.5 * (1.0 + erf((norm(r_grid - R_α) - (r_vdw + r_probe)) / Δ))
end

function cavity_dielectric!(de::DielectricFunction, coords::Vector{SVector{3, Float64}}, labels::Vector{String}, ϵ_solv::Float64, ϵ_vac::Float64)
    """
    Computes the dielectric function defined in
    J. Chem. Phys. 148, 222834 (2018); https://doi.org/10.1063/1.5023916
    We set everything in the cluster volume equal to 1.0 as defined in the paper.
    """
    Threads.@threads for i_x in ProgressBar(1:length(de.grid_x))
        for i_y in 1:length(de.grid_y)
            for i_z in 1:length(de.grid_z)
                r_grid = @SVector [de.grid_x[i_x], de.grid_y[i_y], de.grid_z[i_z]]
                s_α_product = 1.0
                for i_atom in 1:length(labels)
                    s_α_product *= S_α_mSAS(r_grid, coords[i_atom], vdw_radius(labels[i_atom]))
                end
                de.ϵ[i_x, i_y, i_z] = ϵ_vac + (ϵ_solv - ϵ_vac) * s_α_product
            end
        end
    end
    return nothing
end

function cavity_dielectric!(de::DielectricFunction, coords::Vector{SVector{3, Float64}}, labels::Vector{String}, ϵ_solv::Float64, ϵ_vac::Float64, is_in_cavity::Function)
    """
    Computes the dielectric function defined in
    J. Chem. Phys. 148, 222834 (2018); https://doi.org/10.1063/1.5023916
    We set everything in the cluster volume equal to 1.0 as defined in the paper.
    Also uses a function to determine if something is inside of the cavity
    volume. This, gave me bad results however, since they say to use the maximum
    ellipse that can be formed and then set everything inside the ellipse to
    vacuum. This necessarily eliminates much of the structure of the surface and
    just makes it an ellipse. So I'm not sure what they really do...
    """
    Threads.@threads for i_x in ProgressBar(1:length(de.grid_x))
        for i_y in 1:length(de.grid_y)
            for i_z in 1:length(de.grid_z)
                r_grid = @SVector [de.grid_x[i_x], de.grid_y[i_y], de.grid_z[i_z]]
                if is_in_cavity(r_grid) 
                    de.ϵ[i_x, i_y, i_z] = 1.0
                else
                    s_α_product = 1.0
                    for i_atom in 1:length(labels)
                        s_α_product *= S_α_mSAS(r_grid, coords[i_atom], vdw_radius(labels[i_atom]))
                    end
                    de.ϵ[i_x, i_y, i_z] = ϵ_vac + (ϵ_solv - ϵ_vac) * s_α_product
                end
            end
        end
    end
    return nothing
end

function cavity_and_interface_dielectric!(de::DielectricFunction, coords::Vector{SVector{3, Float64}}, labels::Vector{String}, ϵ_solv::Float64, ϵ_vac::Float64, get_ϵ_surr::Function)
    """
    Computes the dielectric function defined in
    J. Chem. Phys. 148, 222834 (2018); https://doi.org/10.1063/1.5023916
    Also includes contribution from an interface where the cluster is located.
    This is included in the same way as for a cavity except that ϵ_solv
    now is evaluated as a function which is tanh fit to the density profile at
    the interface. The function that evluates the interfacial dielectric is
    passed in as a function that takes the grid point and returns ϵ_surr.
    """
    Threads.@threads for i_x in ProgressBar(1:length(de.grid_x))
        for i_y in 1:length(de.grid_y)
            for i_z in 1:length(de.grid_z)
                r_grid = @SVector [de.grid_x[i_x], de.grid_y[i_y], de.grid_z[i_z]]
                s_α_product = 1.0
                for i_atom in 1:length(labels)
                    s_α_product *= S_α_mSAS(r_grid, coords[i_atom], vdw_radius(labels[i_atom]))
                end
                ϵ_surr = get_ϵ_surr(r_grid)
                de.ϵ[i_x, i_y, i_z] = ϵ_vac + (ϵ_surr - ϵ_vac) * s_α_product
            end
        end
    end
    return nothing
end

function interfacial_dielectric(r::SVector{3, Float64}, α::Float64, z_gds::Float64, ϵ_solv::Float64, treat_as_sphere::Bool)
    """
    Calculates the interfacial dielectric function based on a fit to the density
    profile.

    If treat_as_sphere is set to true, then the surface will be treated as if being
    from a spherical droplet. We then assume that the point we are dealing with
    is centered at the origin of the droplet, such that norm(r)-z_gdz will be negative
    if the point is inside the GDS. Otherwise we treat the system as a slab, in which case
    we just compare r_z to z_gds.
    """
    if treat_as_sphere
        #if norm(r) < z_gds
            return 0.5 * ϵ_solv * (1.0 - tanh(α * (norm(r) - z_gds)))
        #end
        #return 0.5 * ϵ_solv * (1.0 + tanh(α * (norm(r) - z_gds)))
    else
        if r[3] < z_gds
            return 0.5 * ϵ_solv * (1.0 + tanh(α * (r[3] - z_gds)))
        end
        return 0.5 * ϵ_solv * (1.0 - tanh(α * (r[3] - z_gds)))
    end
end

function ellipsoid(r::SVector{3, Float64}, r0::SVector{3, Float64}, abc::SVector{3, Float64})
    return sum(((r - r0).^2) ./ (abc.^2))
end

function get_cavity_dielectric_function(coords::Matrix{Float64}, labels::Vector{String}, ϵ_solv::Float64, ϵ_vac::Float64, grid_spacing::Float64=0.2, Δ::Float64=0.265, buffer::Float64=4.0)
    """
    Automatically chooses the grid with spacing given by grid_spacing
    by computing the centroid of the given coordinates and ensuring
    the grid spans the entire cluster plus a buffer into the continuum.
    """
    com = SVector{3, Float64}(center_of_mass(coords, labels))
    a = maximum((coords[1,:] .- com[1]) .+ (vdw_radius.(labels) .- 2 * Δ))
    b = maximum((coords[2,:] .- com[2]) .+ (vdw_radius.(labels) .- 2 * Δ))
    c = maximum((coords[3,:] .- com[3]) .+ (vdw_radius.(labels) .- 2 * Δ))
    abc = SVector{3, Float64}([a, b, c])
    is_in_cavity = r::SVector{3, Float64} -> ellipsoid(r, com, abc) < 1.0

    grid_length = 2.0 * (maximum(abc) + buffer)

    grid_x = LinRange((com[1]-0.5*grid_length):grid_spacing:(com[1]+0.5*grid_length))
    grid_y = LinRange((com[2]-0.5*grid_length):grid_spacing:(com[2]+0.5*grid_length))
    grid_z = LinRange((com[3]-0.5*grid_length):grid_spacing:(com[3]+0.5*grid_length))

	static_coords = [SVector{3, Float64}(coords[:,i]) for i in 1:size(coords, 2)]
    de = DielectricFunction(grid_x, grid_y, grid_z)

    cavity_dielectric!(de, static_coords, labels, ϵ_solv, ϵ_vac)#, is_in_cavity)

    return de
end

function get_interfacial_cavity_dielectric_function(coords::Matrix{Float64}, labels::Vector{String}, ϵ_solv::Float64, ϵ_vac::Float64, α::Float64, z_gds::Float64, treat_as_sphere::Bool, grid_spacing::Float64=0.2, Δ::Float64=0.265, buffer::Float64=4.0)
    """
    Automatically chooses the grid with spacing given by grid_spacing
    by computing the centroid of the given coordinates and ensuring
    the grid spans the entire cluster plus a buffer into the continuum.

    Also takes parameters related to the interface, which is currently assumed
    to be air such that the dielectric is set to ϵ_vac.

    α is the fitting parameter that goes into the tanh function in determining
    the GDS.
    """
    com = SVector{3, Float64}(center_of_mass(coords, labels))
    a = maximum((coords[1,:] .- com[1]) .+ (vdw_radius.(labels) .- 2 * Δ))
    b = maximum((coords[2,:] .- com[2]) .+ (vdw_radius.(labels) .- 2 * Δ))
    c = maximum((coords[3,:] .- com[3]) .+ (vdw_radius.(labels) .- 2 * Δ))
    abc = SVector{3, Float64}([a, b, c])
    get_ϵ_surr = r::SVector{3, Float64} -> interfacial_dielectric(r, α, z_gds, ϵ_solv, treat_as_sphere)

    grid_length = 2.0 * (maximum(abc) + buffer)

    grid_x = LinRange((com[1]-0.5*grid_length):grid_spacing:(com[1]+0.5*grid_length))
    grid_y = LinRange((com[2]-0.5*grid_length):grid_spacing:(com[2]+0.5*grid_length))
    grid_z = LinRange((com[3]-0.5*grid_length):grid_spacing:(com[3]+0.5*grid_length))

	static_coords = [SVector{3, Float64}(coords[:,i]) for i in 1:size(coords, 2)]
    de = DielectricFunction(grid_x, grid_y, grid_z)

    cavity_and_interface_dielectric!(de, static_coords, labels, ϵ_solv, ϵ_vac, get_ϵ_surr)

    return de
end

function plot_dielectric_cross_section(de::DielectricFunction, y_index::Int)
    fig = Figure()

    ax1 = Axis(fig[1,1], fontsize=24)
    
    hm = heatmap!(ax1, de.grid_x, de.grid_z, de.ϵ[:, y_index, :], colormap=:Spectral)
    Colorbar(fig[:, end+1], hm, ticks=5:10:75)
    ax1.xlabel = "x position"
    ax1.ylabel = "z position"
    fig
end
