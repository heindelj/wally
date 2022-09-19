using LinearAlgebra
using StaticArrays
include("units.jl")
include("atomic_masses.jl")
include("molecular_cluster.jl")

function is_a_distance_angle_hydrogen_bond(O_donor_index::Int, H_donor_index::Int, acceptor_index::Int, coords::AbstractMatrix, distance_cutoff::Float64=2.9, angle_cutoff=120.0)
	"""
	Tests if a hydrogen and oxygen are h-bonded based on distance and angle
	cutoffs. Takes the oxygen and hydrogen donor index as well as the acceptor
	index to allow for tests involving hydroxide and hydronium as well.
	"""
	@views bond_vec = coords[:, H_donor_index] - coords[:, O_donor_index]
	@views hbond_vec = coords[:, H_donor_index] - coords[:, acceptor_index]
	angle = acosd(bond_vec ⋅ hbond_vec / (norm(bond_vec) * norm(hbond_vec)))
	if (angle > angle_cutoff && norm(hbond_vec) < distance_cutoff)
		return true
	end	
	return false
end

function number_of_accepted_hydrogen_bonds(coords::AbstractMatrix, labels::AbstractVector{String}, acceptor_index::Int, distance_cutoff::Float64=2.7, angle_cutoff::Float64=115.0)
	"""
	Takes coords, labels, and an index and find the number of hydrogen bonds
	in a system. Cluster must be sorted in OHH order.
	"""
	sorted_labels, sorted_coords = sort_water_cluster(coords, labels)
	num_hbonds = 0
	for i in 1:length(sorted_labels)
		if sorted_labels[i] == "O" && i != acceptor_index
			if is_a_distance_angle_hydrogen_bond(i, i+1, acceptor_index, sorted_coords, distance_cutoff, angle_cutoff)
				num_hbonds += 1
			end
			if is_a_distance_angle_hydrogen_bond(i, i+2, acceptor_index, sorted_coords, distance_cutoff, angle_cutoff)
				num_hbonds += 1
			end
		end
	end
	return num_hbonds
end

function r_psi_hydrogen_bonds(coords::AbstractMatrix)
    """
    Computes the r-psi hydrogen bond of a collection of water molecules, 
    returning which of the atoms is donating a hydrogen bond to the other ones.

    Args:
        coords: 3xN matrix of coordinates in OHH order
    Returns:
        hbonds: dictionary mapping the index of the atoms which donate a hydrogen bond to index of the atom they donate to
    """
    O_indices  = range(1, size(coords, 2), step=3)
    H_indices = sort!([range(2, size(coords, 2), step=3); range(3, size(coords, 2), step=3)])
    hbonds = Dict{Int, Int}()
    for oxygen_index in O_indices
        for hydrogen_index in H_indices
            if !(oxygen_index + 1 == hydrogen_index || oxygen_index + 2 == hydrogen_index)
                if is_a_hydrogen_bond(hydrogen_index, oxygen_index, coords)
                    hbonds[hydrogen_index] = oxygen_index
                end
            end
        end
    end
    return hbonds
end

function is_a_hydrogen_bond(hydrogen_index::Int, oxygen_index::Int, coords::AbstractMatrix)
    """
    Tests if the hydrogen at hydrogen_index is hbonded to oxygen_index. If so, returns true.
    Coords is required to be in OHH order and in angstroms.
    """
    acceptor_OH1 = view(coords, :, oxygen_index) - view(coords, :, oxygen_index+1)
    acceptor_OH2 = view(coords, :, oxygen_index) - view(coords, :, oxygen_index+2)
    water_normal = cross(acceptor_OH1, acceptor_OH2)
    r = view(coords, :, hydrogen_index) - view(coords, :, oxygen_index)

    ψ = acosd(dot(water_normal, r) / (norm(water_normal) * norm(r)))
    if ψ > 90.0
        ψ = 180.0 - ψ
    end
    
    N = exp(-norm(r) / 0.343) * (7.1 - 0.05 * ψ + 0.00021 * ψ^2)
    return N > 0.0085
end

function number_of_hydrogen_bonds(coords::AbstractMatrix)
    """
    returns the number of hydrogen bonds in an OHH-sorted water system.
    """
    return length(r_psi_hydrogen_bonds(coords))
end

function number_of_hydrogen_bonds(coords::Vector{Matrix{Float64}})
    """
    returns the number of hydrogen bonds in a sequence of OHH-sorted water systems.
    """
    number_of_hydrogen_bonds_in_each_frame = zeros(Int, length(coords))
    for (i, coord) in enumerate(coords)
        number_of_hydrogen_bonds_in_each_frame[i] = number_of_hydrogen_bonds(coord)
    end
    return number_of_hydrogen_bonds_in_each_frame
end

function sort_water_cluster(coords::AbstractMatrix, labels::AbstractVector, to_angstrom::Bool = false; return_permutation::Bool = false)
    """
    Sorts water clusters containing water, hydroxide, or hydronium
	and puts H3O+ and OH- first. Then OHH sorted water. Then any other
    solutes in the system.
	"""
    O_neighbors = Dict{Int, Vector{Int}}()
    for i in eachindex(labels)
        if labels[i] == "O"
            O_neighbors[i] = []
        end
    end

	if to_angstrom
		coords *= conversion(:angstrom, :bohr)
    end

    # build neighbor list of atoms.
    nl = KDTree(coords)
    # assume ten nearest atoms contains the nearest oxygen
    indices, _ = knn(nl, coords, 10, true)

    # associate each hydrogen to the closest oxygen
    for nbr_indices in indices
        if labels[nbr_indices[1]] == "H"
            for i in nbr_indices
                if labels[i] == "O"
                    push!(O_neighbors[i], nbr_indices[1])
                    break
                end
            end
        end
    end

    OH_indices = Int[]
    hydronium_indices = Int[]
    water_indices = Int[]
    all_indices = Int[]
    for i in keys(O_neighbors)
        if length(O_neighbors[i]) == 1
            push!(OH_indices, i)
            append!(OH_indices, O_neighbors[i])
        elseif length(O_neighbors[i]) == 2
            push!(water_indices, i)
            append!(water_indices, O_neighbors[i])
        elseif length(O_neighbors[i]) == 3
            push!(hydronium_indices, i)
            append!(hydronium_indices, O_neighbors[i])
        else
            println(i)
            display(O_neighbors[i])
            @assert false "Found a water with four hydrogen atoms or no hydrogen atoms. Exiting."
        end
    end

    append!(all_indices, OH_indices)
    append!(all_indices, hydronium_indices)
    append!(all_indices, water_indices)
	
    unused_indices = setdiff([1:length(labels)...], all_indices)
    append!(all_indices, unused_indices)
    @assert length(all_indices) == length(labels) "Didn't use an index or used an index twice. Bug somewhere."
	if return_permutation
		return all_indices
	end
	return labels[all_indices], coords[:, all_indices]
end

function get_n_neighboring_waters(coords::AbstractMatrix, labels::AbstractVector, special_index::Int, num_neighbors::Int)
    @assert labels[special_index] == "O" "Currently only support using an oxygen as the special index."
	static_coords = [SVector{3, Float64}(coords[:,i]) for i in 1:size(coords, 2)]
	distances = zeros(count(==("O"), labels))
	distance_indices = zeros(Int, count(==("O"), labels))
	dist_index = 1
	@inbounds for i in 1:length(labels)
		if labels[i] == "O"
			if i != special_index
                @views distances[dist_index] = norm(static_coords[i] - static_coords[special_index])
				distance_indices[dist_index] = i
	        	dist_index += 1
			else
				distances[dist_index] = 100000000.0 # avoid choosing special atom as neighbor of itself
				distance_indices[dist_index] = i
				dist_index += 1
			end
	    end
    end
	permuted_indices = distance_indices[partialsortperm(distances, 1:num_neighbors)]

	# now find the hydrogens bonded to the nearest neighbors
	final_indices = Int[]

	temp_indices = Int[]
	@inbounds for index in [special_index, permuted_indices...]
		push!(final_indices, index)
        for i in 1:length(labels)
		    if labels[i] == "H"
				@views if (static_coords[index] - static_coords[i])⋅(static_coords[index] - static_coords[i]) < 1.3*1.3
                    if i ∉ final_indices
				        push!(temp_indices, i)
                    end
                    if length(temp_indices) > 2
                        # we've found too many close hydrogens
                        OH_1 = 100000000000.0
                        OH_2 = 100000000000.0
                        OH_index_1 = 0
                        OH_index_2 = 0
                        for H_index in temp_indices
                            dist = norm(static_coords[index] - static_coords[H_index])
                            if dist < OH_1
                                OH_1 = dist
                                OH_index_1 = H_index
                            elseif dist < OH_2
                                OH_2 = dist
                                OH_index_2 = H_index
                            end
                        end
                        @assert OH_index_1 > 0 "Failed to sort out too many H neighbors problem."
                        @assert OH_index_2 > 0 "Failed to sort out too many H neighbors problem."
                        empty!(temp_indices)
                        push!(temp_indices, H_index_1)
                        push!(temp_indices, H_index_2)
                    end
		 	    end
		    end
	    end
		append!(final_indices, temp_indices)
		empty!(temp_indices)
    end
	return labels[final_indices], coords[:, final_indices]
end

function get_hydroxide_indices(coords::AbstractMatrix, labels::AbstractVector{String}; max_matches::Int=typemax(Int))
	"""
	Finds the hydroxide ions in a collection of Hs and Os.
	max_matches means to stop after finding a certain number of OH-.
	"""
	static_coords = [SVector{3, Float64}(coords[:,i]) for i in 1:size(coords, 2)]
	water_indices = Int[]
	hydroxide_indices = Int[]
	O_indices = zeros(Int, count(==("O"), labels))
	H_indices = zeros(Int, count(==("H"), labels))
	
	# get indices of each atom type
	O_counter = 1
	H_counter = 1
	for i in 1:length(labels)
		if labels[i] == "O" || labels[i] == "o"
			O_indices[O_counter] = i
			O_counter += 1
		elseif labels[i] == "H" || labels[i] == "h"
			H_indices[H_counter] = i
			H_counter += 1
	    else
		    @assert false "Atom other than O or H. Fix the code to handle this case lazy guy."
	    end
	end
	
	@assert length(H_indices) == H_counter-1
	@assert length(O_indices) == O_counter-1
	@assert length(labels) == size(coords, 2) "Not the same number of labels as coordinates."
	@inbounds for O_index in O_indices
		push!(water_indices, O_index)
		for H_index in H_indices
			@views if (static_coords[O_index] - static_coords[H_index])⋅(static_coords[O_index] - static_coords[H_index]) < 1.3 * 1.3
				push!(water_indices, H_index)
				if length(water_indices) > 2 # not a hydroxide
					break
				end
			end
		end
		if length(water_indices) == 2
			append!(hydroxide_indices, water_indices)
		end
		if length(hydroxide_indices) == 2 * max_matches
			break
		end
		empty!(water_indices)
	end

	return hydroxide_indices
end

function has_hemibonded_structure(coords::AbstractMatrix{T}, labels::AbstractVector{String}, sort_cluster::Bool=true) where T <: Real
    """
    Determines if a hydroxide water cluster has any hemibonded structures
    based on the geometric criteria described in: 
    Chipman, D. M. (2011). Hemibonding between hydroxyl radical and water.
    The Journal of Physical Chemistry A, 115(7), 1161-1171.

    Sorts the water cluster and finds all hydroxide indices then
    checks for hemibonds against all other waters.

    Returns -1 if no hemibonded structure found and the index of oxygen
    in water which is hemibonded if such a structure is found.
    """
    if sort_cluster
        labels, coords = sort_water_cluster(coords, labels)
    end
    hydroxide_indices = get_hydroxide_indices(coords, labels)
    sanity_check = ((size(coords, 2) - length(hydroxide_indices)) % 3 == 0) 
    if (!sanity_check)
        println("Warning: After accounting for hydroxide, this is not pure water. Have to know how to neglect the other atoms or better classify the hydroxide.")
        println("Hydroxide indices: ", hydroxide_indices)
        println("Num atoms total: ", size(coords, 2))
        println("RETURNING FALSE!")
        return -1
    end

    water_indices = setdiff([1:length(labels)...], hydroxide_indices)
    water_array = get_array_of_waters(coords[:, water_indices], labels[water_indices])
    for i in 1:2:length(hydroxide_indices)
        hydroxide_O_index = hydroxide_indices[i]
        for (j, water_geom) in enumerate(water_array)
            OO_vec = coords[:,i] - water_geom[:,1]
            OH_vec = coords[:,i] - coords[:,i+1]
            HOH_bisector = 0.5 * ((water_geom[:,1]-water_geom[:,2]) + (water_geom[:,1]-water_geom[:,3]))
            α = acosd(OO_vec ⋅ OH_vec / (norm(OO_vec) * norm(OH_vec)))
            r_OO = norm(OO_vec)
            χ = acosd(HOH_bisector ⋅ OO_vec / (norm(OO_vec) * norm(HOH_bisector)))
            if r_OO > 2.38 && r_OO < 2.92
                if α > 52.0 && α < 137.0
                    if χ > 20.0 && χ < 60.0
                        return length(hydroxide_indices) + 3*(j-1) + 1
                    end
                end
            end
        end
    end
    return -1
end

function sort_waters!(coords::AbstractArray{Matrix{T}, 1}, labels::AbstractVector{Vector{String}}; to_angstrom::Bool = false) where T <: Real
    Threads.@threads for i in 1:length(coords)
        coords[i] = sort_waters(coords[i], labels[i], to_angstrom=to_angstrom)
    end
end

function get_array_of_waters(coords::AbstractMatrix, labels::AbstractVector; to_angstrom::Bool = false)
    """
    Returns an array of arrays where each element is 3x3 array containing a water molecule.
    """
    @assert isinteger(size(coords, 2) / 3) "Number of atoms not divisble by 3. Is this water?"
    new_coords = Array{typeof(coords), 1}(undef, size(coords, 2) ÷ 3)
    coords = sort_waters(coords, labels, to_angstrom=to_angstrom)

    for i in 1:(size(coords, 2) ÷ 3)
        new_coords[i] = coords[:, ((i-1)*3 + 1):(i*3)]
    end
    return new_coords
end

function get_array_of_waters(coords::AbstractMatrix; to_angstrom::Bool = false)
    """
    Returns an array of arrays where each element is 3x3 array containing a water molecule.
    """
    @assert isinteger(size(coords, 2) / 3) "Number of atoms not divisble by 3. Is this water?"
    new_coords = Array{typeof(coords), 1}(undef, size(coords, 2) ÷ 3)
    coords = sort_waters(coords, to_angstrom=to_angstrom)

    for i in 1:(size(coords, 2) ÷ 3)
        new_coords[i] = coords[:, ((i-1)*3 + 1):(i*3)]
    end
    return new_coords
end

function sort_water_molecules_to_oxygens_first(coords::AbstractMatrix)
    """
    Sorts waters in OHHOHH order to OOHHHH order.
    """
    new_coords = zero(coords)
    j::Int=1
    Nw::Int=div(size(coords, 2), 3)
    for i = 1:Nw
        @inbounds @views new_coords[:,i] = coords[:,(i-1)*3+1]
        @inbounds @views new_coords[:,Nw+j] = coords[:,(i-1)*3+2]
        @inbounds @views new_coords[:,Nw+j+1] = coords[:,(i-1)*3+3]
        j += 2
    end
    return new_coords
end

function sort_water_molecules_to_oxygens_first!(new_coords::AbstractMatrix, coords::AbstractMatrix)
    """
    Sorts waters in OHHOHH order to OOHHHH order.
    """
    j::Int=1
    Nw::Int=div(size(coords, 2), 3)
    for i = 1:Nw
        @inbounds @views new_coords[:,i] = coords[:,(i-1)*3+1]
        @inbounds @views new_coords[:,Nw+j] = coords[:,(i-1)*3+2]
        @inbounds @views new_coords[:,Nw+j+1] = coords[:,(i-1)*3+3]
        j += 2
    end
    return new_coords
end

function sort_oxygens_first_to_water_molecules(coords::AbstractMatrix)
    """
    Sorts waters in OOHHHH order to OHHOHH order.
    """
    new_coords = zero(coords)
    j::Int=1
    Nw::Int=div(size(coords, 2), 3)
    for i = 1:Nw
        @inbounds @views new_coords[:,(i-1)*3+1] = coords[:,i]
        @inbounds @views new_coords[:,(i-1)*3+2] = coords[:,Nw+j]
        @inbounds @views new_coords[:,(i-1)*3+3] = coords[:,Nw+j+1]
        j += 2
    end
    return new_coords
end
