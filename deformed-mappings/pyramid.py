from newton_generation import *


class LinearPyramid(LinearBase):
    def __init__(self):

        num_vertices = 5
        ndim = 3
        name = "linear_3d"
        namespace = "Pyramid"
        x_description = """

max_width = -xi_2 + 1.0
b0 = ((xi0 + 1.0) / max_width) * 2.0 - 1.0
b1 = ((xi1 + 1.0) / max_width) * 2.0 - 1.0
a0 = 0.5 * (v4 - v0) * (xi_2 + 1.0) + v0
a1 = 0.5 * (v4 - v1) * (xi_2 + 1.0) + v1
a2 = 0.5 * (v4 - v2) * (xi_2 + 1.0) + v2
a3 = 0.5 * (v4 - v3) * (xi_2 + 1.0) + v3

X(xi) = 0.25 * a0 * (1 - b0) * (1 - b1) +
        0.25 * a1 * (1 + b0) * (1 - b1) +
        0.25 * a2 * (1 + b0) * (1 + b1) +
        0.25 * a3 * (1 - b0) * (1 + b1)
"""
        LinearBase.__init__(self, num_vertices, ndim, name, namespace, x_description)

    def get_x(self, xi):

        v = self.vertices

        max_width = -xi[2] + 1.0
        b0 = ((xi[0] + 1.0) / max_width) * 2.0 - 1.0
        b1 = ((xi[1] + 1.0) / max_width) * 2.0 - 1.0
        a0 = 0.5 * (v[4] - v[0]) * (xi[2] + 1.0) + v[0]
        a1 = 0.5 * (v[4] - v[1]) * (xi[2] + 1.0) + v[1]
        a2 = 0.5 * (v[4] - v[2]) * (xi[2] + 1.0) + v[2]
        a3 = 0.5 * (v[4] - v[3]) * (xi[2] + 1.0) + v[3]

        x = (
            0.25 * a0 * (1 - b0) * (1 - b1)
            + 0.25 * a1 * (1 + b0) * (1 - b1)
            + 0.25 * a2 * (1 + b0) * (1 + b1)
            + 0.25 * a3 * (1 - b0) * (1 + b1)
        )

        return x


def self_test():

    geom_x = LinearPyramid()

    vertices_ref = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )
    geom_ref = LinearGeomEvaluate(geom_x, vertices_ref)

    for vx in vertices_ref[:-1]:
        to_test = geom_ref.x(vx)
        correct = vx
        assert np.linalg.norm(np.array(correct).ravel() - np.array(to_test).ravel(), np.inf) < 1.0e-15
        to_test = geom_ref.f(vx, vx)
        assert np.linalg.norm(np.array(to_test).ravel(), np.inf) < 1.0e-15

    vertices_test = (
        (-3.0, -2.0, 2.0),
        (1.0, -1.0, 2.0),
        (2.0, 2.0, 2.5),
        (-1.0, 4.0, 4.5),
        (-3.0, -2.0, -0.1),
    )
    geom_test = LinearGeomEvaluate(geom_x, vertices_test)
    geom_newton = Newton(geom_x)

    geom_newton_evaluate = NewtonEvaluate(geom_newton, geom_test)

    xi_correct0 = -0.9
    xi_correct1 = 0.8
    xi_correct2 = 0.2
    xi_correct = (xi_correct0, xi_correct1, xi_correct2)
    phys = geom_test.x(xi_correct)
    residual, fv = geom_newton_evaluate.residual(xi_correct, phys)
    assert residual < 1.0e-15

    xi = [0.0, 0.0, 0.0]
    for stepx in range(5):
        residual, fv = geom_newton_evaluate.residual(xi, phys)
        xin = geom_newton_evaluate.step(xi, phys, fv)
        xi[0] = xin[0]
        xi[1] = xin[1]
        xi[2] = xin[2]

    assert abs(xi[0] - xi_correct[0]) < 1.0e-14
    assert abs(xi[1] - xi_correct[1]) < 1.0e-14
    assert abs(xi[2] - xi_correct[2]) < 1.0e-14

    vertices_test = (
        (-1, -1, -0.6),
        (-0.8, -1, -0.6),
        (-0.8, -0.8, -0.6),
        (-1, -0.8, -0.6),
        (-0.9, -0.9, -0.54),
    )
    geom_test = LinearGeomEvaluate(geom_x, vertices_test)
    geom_newton = Newton(geom_x)

    geom_newton_evaluate = NewtonEvaluate(geom_newton, geom_test)

    for vi, vx in enumerate(vertices_ref[:-1]):
        to_test = geom_test.x(vx)
        correct = vertices_test[vi]
        assert np.linalg.norm(np.array(correct).ravel() - np.array(to_test).ravel(), np.inf) < 1.0e-15
        to_test = geom_test.f(vx, correct)
        assert np.linalg.norm(np.array(to_test).ravel(), np.inf) < 1.0e-15

    xi_correct0 = -0.6
    xi_correct1 = -0.5
    xi_correct2 = -0.2

    xi_correct = (xi_correct0, xi_correct1, xi_correct2)
    phys = geom_test.x(xi_correct)
    residual, fv = geom_newton_evaluate.residual(xi_correct, phys)
    assert residual < 1.0e-15

    # phys_nektar = ( -0.960000, -0.990000, -0.030000)
    phys_nektar = (-0.920000, -0.910000, -0.576000)

    assert abs(phys_nektar[0] - phys[0]) < 1.0e-6
    assert abs(phys_nektar[1] - phys[1]) < 1.0e-6
    assert abs(phys_nektar[2] - phys[2]) < 1.0e-6

    xi = [0.0, 0.0, 0.0]
    for stepx in range(5):
        residual, fv = geom_newton_evaluate.residual(xi, phys)
        xin = geom_newton_evaluate.step(xi, phys, fv)
        xi[0] = xin[0]
        xi[1] = xin[1]
        xi[2] = xin[2]

    assert abs(xi[0] - xi_correct[0]) < 1.0e-14
    assert abs(xi[1] - xi_correct[1]) < 1.0e-14
    assert abs(xi[2] - xi_correct[2]) < 1.0e-14


def get_type():
    self_test()
    return LinearPyramid


if __name__ == "__main__":
    self_test()
