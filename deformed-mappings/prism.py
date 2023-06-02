from newton_generation import *
"""
PRISM
nverts 6
0 ( -1 , -1 , -5.31519e-13 ),
1 ( -0.8 , -1 , -5.31533e-13 ),
2 ( -0.8 , -1 , -0.2 ),
3 ( -1 , -1 , -0.2 ),
4 ( -1 , -0.8 , -5.31542e-13 ),
5 ( -1 , -0.8 , -0.2 ),
-1 -1 -1 	| -1 -1 -5.31519e-13    -> v0
1 -1 -1 	| -0.8 -1 -5.31533e-13  -> v1
1 1 -1 	| -0.8 -1 -0.2              -> v2
-1 1 -1 	| -1 -1 -0.2            -> v3
-1 -1 1 	| -1 -0.8 -5.31542e-13  -> v4
1 -1 1 	| -nan -nan -nan
1 1 1 	| -nan -nan -nan
-1 1 1 	| -1 -0.8 -0.2              -> v5
test coord = ( -0.6 , -0.7 , -0.9 )
correct_global_coord = ( -0.960000, -0.990000, -0.030000)
"""


class LinearPrism(LinearBase):
    def __init__(self):

        num_vertices = 6
        ndim = 3
        name = "linear_3d"
        x_description = """
        TODO
"""
        LinearBase.__init__(self, num_vertices, ndim, name, x_description)

    def get_x(self, xi):

        v = self.vertices

        a0 = 0.5 * (v[3] - v[0]) * (xi[1] + 1.0) + v[0]
        a1 = 0.5 * (v[2] - v[1]) * (xi[1] + 1.0) + v[1]
        a4 = 0.5 * (v[5] - v[4]) * (xi[1] + 1.0) + v[4]

        A = 0.5 * Matrix([
            [a1[0] - a0[0], a4[0] - a0[0]],
            [a1[1] - a0[1], a4[1] - a0[1]],
            [a1[2] - a0[2], a4[2] - a0[2]]
        ])

        s = Matrix([-1, -1])
        xi_t = Matrix([xi[0], xi[2]])

        x = A @ (xi_t - s) + a0

        return x


if __name__ == "__main__":

    geom_x = LinearPrism()

    vertices_ref = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
    )
    geom_ref = LinearGeomEvaluate(geom_x, vertices_ref)

    for vx in vertices_ref:
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
        (-1.0, 4.0, -1.5),
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

    vertices_test = (
        ( -1 , -1 , 0.0 ),
        ( -0.8 , -1 , 0.0 ),
        ( -0.8 , -1 , -0.2 ),
        ( -1 , -1 , -0.2 ),
        ( -1 , -0.8 , 0.0 ),
        ( -1 , -0.8 , -0.2 ),
    )
    geom_test = LinearGeomEvaluate(geom_x, vertices_test)
    geom_newton = Newton(geom_x)

    geom_newton_evaluate = NewtonEvaluate(geom_newton, geom_test)

    for vi, vx in enumerate(vertices_ref):
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
    phys_nektar = ( -0.960000, -0.920000, -0.050000)

    assert abs(phys_nektar[0] - phys[0]) < 1.0e-8
    assert abs(phys_nektar[1] - phys[1]) < 1.0e-8
    assert abs(phys_nektar[2] - phys[2]) < 1.0e-8

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

    geom_newton_evaluate = NewtonLinearCCode(geom_newton)
    print(geom_newton_evaluate.residual())
    print(geom_newton_evaluate.step())
