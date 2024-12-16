import gmsh
import sys
import numpy as np


if __name__ == "__main__":

    gmsh.initialize()
    gmsh.open(sys.argv[1])
    element_order = int(sys.argv[2])
    perturb_max = 0.02

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(element_order)

    def get_perturbation(shape):
        return np.random.uniform(-perturb_max, perturb_max, size=shape)

    def on_boundary(vertex):
        v0 = abs(abs(vertex[0]) - 1) < 1e-8
        v1 = abs(abs(vertex[1]) - 1) < 1e-8
        return v0 or v1

    def perturb_node(n):
        v = n[0]
        if not on_boundary(v):
            t = get_perturbation(v.shape)
            t[2] = 0.0
            v += t
        return v, n[1], n[2], n[3]

    vertex_ids = set(gmsh.model.mesh.get_nodes()[0])
    for vx in vertex_ids:
        n = gmsh.model.mesh.get_node(vx)
        print(n)
        nn = perturb_node(n)
        gmsh.model.mesh.set_node(vx, nn[0], nn[1])

    order = f"_order_{element_order}" if element_order > 1 else ""
    filename = f"mixed_ref_square{order}.msh"
    gmsh.write(filename)

    if "--visualise" in sys.argv:
        gmsh.fltk.run()


