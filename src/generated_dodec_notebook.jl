### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 77db2f67-8648-4744-9501-2923e90e8739
using Bio3DView

# ╔═╡ 94f05c30-b0d1-41ec-a151-4285e355b847
include("water_cluster_graphs/src/tcode_to_cluster_structure.jl")

# ╔═╡ 5d09cf3c-988a-11eb-1f6b-a18fe1615d24
cd("/home/heindelj/Coding_Projects/julia_development/wally/src/")

# ╔═╡ 9786d3f0-335f-4483-ad9b-45ae96b36195
dodec_geom = construct_dodecahedral_cage(2.65)

# ╔═╡ 72e07f8b-16e9-4f45-b582-d4129750792e
tcodes = load_tcode_file("water_cluster_graphs/src/data/dodecahedron_directed_graphs.tcode")

# ╔═╡ 40444717-b7c5-4f40-8e4d-244aa673bd2e
generated_dodec_geom = structure_from_tcode(tcodes[1])

# ╔═╡ 2f067d9c-ab0c-4095-b185-eaa8ed75b5f1
header, labels, ref_geoms = read_xyz("water_cluster_graphs/src/data/w20_dodecahedron_reference.xyz")

# ╔═╡ 6b923f2c-8aac-4893-9a8e-bd4e81e62e24
write_xyz("temp.xyz", header, labels, [generated_dodec_geom])

# ╔═╡ 90f720f8-602b-475c-a480-8f5964015376
viewfile("temp.xyz", "xyz")

# ╔═╡ e522ba2d-d806-4a29-b99d-9d59ed12ccd9
rm("temp.xyz")

# ╔═╡ 043b6a6b-97f4-451f-9156-a963866958b3


# ╔═╡ Cell order:
# ╠═77db2f67-8648-4744-9501-2923e90e8739
# ╠═5d09cf3c-988a-11eb-1f6b-a18fe1615d24
# ╠═94f05c30-b0d1-41ec-a151-4285e355b847
# ╠═9786d3f0-335f-4483-ad9b-45ae96b36195
# ╠═72e07f8b-16e9-4f45-b582-d4129750792e
# ╠═40444717-b7c5-4f40-8e4d-244aa673bd2e
# ╠═2f067d9c-ab0c-4095-b185-eaa8ed75b5f1
# ╠═6b923f2c-8aac-4893-9a8e-bd4e81e62e24
# ╠═90f720f8-602b-475c-a480-8f5964015376
# ╠═e522ba2d-d806-4a29-b99d-9d59ed12ccd9
# ╠═043b6a6b-97f4-451f-9156-a963866958b3
