using LinearAlgebra
include("units.jl")
include("atomic_masses.jl")

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

function number_of_hydrogen_bonds(coords::Array{Array{Float64, 2}, 1})
    """
    returns the number of hydrogen bonds in a sequence of OHH-sorted water systems.
    """
    number_of_hydrogen_bonds_in_each_frame = zeros(Int, length(coords))
    for (i, coord) in enumerate(coords)
        number_of_hydrogen_bonds_in_each_frame[i] = number_of_hydrogen_bonds(coord)
    end
    return number_of_hydrogen_bonds_in_each_frame
end

function sort_waters(coords::AbstractMatrix; to_angstrom::Bool = false)
    """
    Sorts waters based on distance criteria into OHH order.
    """
    @assert isinteger(size(coords, 2) / 3) "Number of atoms not divisible by 3. Can't be only waters."
    distance_condition::Float64 = 1.4 # ansgtroms
    if to_angstrom
        distance_condition *= conversion(:angstrom, :bohr)
    end

    sorted_indices = zeros(Int, size(coords, 2))
    current_index = 1

    water_indices::Vector{Int} = [0, 0, 0]
    pairs::Vector{Vector{Tuple{Int, Int}}} = [[(1,2), (1,3)], [(2,1), (2,3)], [(3,1), (3,2)]]
    for i in 1:size(coords, 2)
        if !(i in sorted_indices)
            water_indices[1] = i
            count::Int = 2
                for j in 1:size(coords, 2)
                    @inbounds if norm(@view(coords[:,i]) - @view(coords[:,j])) < distance_condition && norm(@view(coords[:,i]) - @view(coords[:,j])) > 0.0001
                        water_indices[count] = j
                        count += 1
                    end
                end

            if count == 4 # otherwise we started with a hydrogen
                for pair in pairs
                    # this is an indexing abomination
                    @inbounds @views OH1 = coords[:, water_indices[pair[1][1]]] - coords[:, water_indices[pair[1][2]]]
                    @inbounds @views OH2 = coords[:, water_indices[pair[2][1]]] - coords[:, water_indices[pair[2][2]]]
                    angle = acosd(dot(OH1, OH2) / (norm(OH1) * norm(OH2)))
                    if angle > 75.0 && angle < 135.0
                        sorted_indices[current_index] = water_indices[pair[1][1]]
                        sorted_indices[current_index+1] = water_indices[pair[1][2]]
                        sorted_indices[current_index+2] = water_indices[pair[2][2]]
                        current_index += 3
                    end
                end
            end
            
        end
    end
    @assert length(sorted_indices) == size(coords, 2) "Didn't associate every atom to a molecule. Check your units (should be in angstroms) or try providing atom labels."
    return coords[:, sorted_indices]
end

function sort_waters(coords::AbstractMatrix, labels::AbstractVector; to_angstrom::Bool = false)
    """
    Sorts waters based on distance criteria into OHH order.
    Takes the associated labels for simplicity of determining which atoms are oxygens.
    """
    @assert isinteger(length(labels) / 3) "Number of atoms not divisible by 3. Can't be only waters."
    
    distance_condition::Float64 = 1.4 # ansgtroms
    if to_angstrom
        distance_condition *= conversion(:angstrom, :bohr)
    end

    sorted_indices = zeros(Int, length(labels))
    oxygen_indices::Array{Int} = zeros(Int, length(labels) ÷ 3)
    hydrogen_indices::Array{Int} = zeros(Int, 2 * (length(labels) ÷ 3))
    # get the oxygen and hydrogen indices
    O_counter::Int = 0
    H_counter::Int = 0
    for (i, label) in enumerate(labels)
        if label == "O" || label == "o"
            O_counter += 1
            oxygen_indices[O_counter] = i
            sorted_indices[3*O_counter-2] = i
        elseif label == "H" || label == "h"
            H_counter += 1
            hydrogen_indices[H_counter] = i
        end
    end
    for (i_Oxygen, O_vec) in enumerate(eachcol(view(coords, :, oxygen_indices)))
        num_H_found::Int = 1
        for (i_Hydrogen, H_vec) in enumerate(eachcol(view(coords, :, hydrogen_indices)))
            if (norm(O_vec - H_vec) < distance_condition)
                sorted_indices[3*(i_Oxygen - 1) + num_H_found + 1] = hydrogen_indices[i_Hydrogen]
                num_H_found += 1
            end
        end
    end

    return coords[:, sorted_indices]
end

function sort_water_cluster(coords::AbstractMatrix, labels::AbstractVector, to_angstrom::Bool = false)
    """
    Sorts water clusters containing water, hydroxide, or hydronium
	and puts H3O+ and OH- first. Then OHH sorted water.
	"""
    sorted_coords = zero(coords)
    O_indices = Int[]
	H_indices = Int[]

	sorted_hydroxide_indices = Int[]
	sorted_water_indices = Int[]
	sorted_hydronium_indices = Int[]

	distance_condition = 1.4
	if to_angstrom
		distance_condition *= conversion(:angstrom, :bohr)
    end

	# get indices of each atom type
	for i in 1:length(labels)
		if labels[i] == "O" || labels[i] == "o"
		    push!(O_indices, i)
		elseif labels[i] == "H" || labels[i] == "h"
		    push!(H_indices, i)
	    else
		    @assert false "Atom other than O or H. Fix the code to handle this case lazy guy."
	    end
	end

	indices_temp = Int[]
    for O_index in O_indices
        @views O_vec = coords[:, O_index]
		push!(indices_temp, O_index)
		for H_index in H_indices
		    @views H_vec = coords[:, H_index]
			if norm(O_vec - H_vec) < distance_condition
                push!(indices_temp, H_index)
		    end
        end

		if length(indices_temp) == 2
		    append!(sorted_hydroxide_indices, indices_temp)
	    elseif length(indices_temp) == 3
		    append!(sorted_water_indices, indices_temp)
		elseif length(indices_temp) == 4
		    append!(sorted_hydronium_indices, indices_temp)
	    else
		    @assert false "Found more than four hydrogen neighbors to an oxygen. Quitting. Fix your code."
		    # need to handle the case that there are too many
			# neighbors which shouldn't happen but probably could
	    end
		empty!(indices_temp)
    end

	append!(sorted_hydroxide_indices, sorted_hydronium_indices)
	append!(sorted_hydroxide_indices, sorted_water_indices)

	return labels[sorted_hydroxide_indices], coords[:, sorted_hydroxide_indices]
end

function get_n_neighboring_waters(coords::AbstractMatrix, labels::AbstractVector, special_index::Int, num_neighbors::Int)
    @assert labels[special_index] == "O" "Currently only support using an oxygen as the special index."
	distances = zeros(count(==("O"), labels))
	distance_indices = zeros(Int, count(==("O"), labels))
	dist_index = 1
	for i in 1:length(labels)
		if labels[i] == "O"
		    @views distances[dist_index] = norm(coords[:, i] - coords[:, special_index])
			distance_indices[dist_index] = i
	        dist_index += 1
	    end
    end
	distances[special_index] = 100000000.0 # avoid choosing special atom as neighbor of itself
	permuted_indices = distance_indices[partialsortperm(distances, 1:num_neighbors)]

	# now find the hydrogens bonded to the nearest neighbors
	final_indices = Int[]

	temp_indices = Int[]
	for index in [special_index, permuted_indices...]
		push!(final_indices, index)
        for i in 1:length(labels)
		    if labels[i] == "H"
				if norm(coords[:, index] - coords[:, i]) < 1.4
				    push!(temp_indices, i)
		 	    end
		    end
	    end
		append!(final_indices, temp_indices)
		empty!(temp_indices)
    end
	return labels[final_indices], coords[:, final_indices]
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
