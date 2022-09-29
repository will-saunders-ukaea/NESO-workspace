import pygmsh as gmsh


lc = 0.10
E = 1.0
N_stripes = 2
width = 1.0 / (float(N_stripes) * 2.0)

stripes_triangles = []
stripes_quads = []

with gmsh.geo.Geometry() as geo:

    for stripex in range(N_stripes):
        stripes_triangles.append(
            geo.add_polygon(
                [
                    [0.0, stripex * width * 2],
                    [E, stripex * width * 2],
                    [E, stripex * width * 2 + width],
                    [0.0, stripex * width * 2 + width],
                ],
                mesh_size=lc,
            )
        )
        stripes_quads.append(
            geo.add_polygon(
                [
                    [0.0, stripex * width * 2 + width],
                    [E, stripex * width * 2 + width],
                    [E, stripex * width * 2 + width * 2],
                    [0.0, stripex * width * 2 + width * 2],
                ],
                mesh_size=lc,
            )
        )

    geo.add_physical(stripes_triangles, "triangles")
    geo.add_physical(stripes_quads, "quads")

    geo.add_raw_code('Mesh.RecombinationAlgorithm=2;\n')

    #mesh = geo.generate_mesh(dim=2, algorithm=8)
    mesh = geo.generate_mesh(dim=2)

    import pdb; pdb.set_trace()

    mesh.write(f"mesh_mixed_{lc}.vtu")
    gmsh.write(f"mesh_mixed_{lc}.msh")
