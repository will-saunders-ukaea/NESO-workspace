from sympy import *
import numpy as np
init_printing(use_unicode=True)


xi0 = symbols("xi0")
xi1 = symbols("xi1")
# xi2 = symbols("xi2")
xi = Matrix([xi0, xi1])

v00 = symbols("v00")
v01 = symbols("v01")
v10 = symbols("v10")
v11 = symbols("v11")
v20 = symbols("v20")
v21 = symbols("v21")
v30 = symbols("v30")
v31 = symbols("v31")

xin0 = symbols("xin0")
xin1 = symbols("xin1")
xin = Matrix([xin0, xin1])

phys0 = symbols("phys0")
phys1 = symbols("phys1")
phys = Matrix(
    [
        phys0,
        phys1
    ]
)


def get_x():
    x = Matrix(
        [
            0.25 * v00 * (1 - xi0) * (1 - xi1) + \
            0.25 * v10 * (1 + xi0) * (1 - xi1) + \
            0.25 * v30 * (1 - xi0) * (1 + xi1) + \
            0.25 * v20 * (1 + xi0) * (1 + xi1),
            0.25 * v01 * (1 - xi0) * (1 - xi1) + \
            0.25 * v11 * (1 + xi0) * (1 - xi1) + \
            0.25 * v31 * (1 - xi0) * (1 + xi1) + \
            0.25 * v21 * (1 + xi0) * (1 + xi1)
        ]
    )
    return x

def newton_step(vertices, a0, a1, phys0_val, phys1_val):

    subs = {
        xi0 :  a0,
        xi1 :  a1,
        phys0: phys0_val,
        phys1: phys1_val,
    }
    subs.update(vertices)

    xin0v = step_list[0].evalf(subs=subs)
    xin1v = step_list[1].evalf(subs=subs)
    return xin0v, xin1v

def xi_to_phys(vertices, xi0v, xi1v):

    subs = {
        xi0 :  xi0v,
        xi1 :  xi1v
    }
    subs.update(vertices)

    physv = x.evalf(subs=subs)
    return physv


def newton_residual(vertices, xi0, xi1, phys0v, phys1v):
    target = np.array((phys0v, phys1v)).ravel()
    out = np.array(xi_to_phys(vertices, xi0, xi1)).ravel()
    return np.linalg.norm(target - out, np.inf)



def print_x(step):

    print(
"""
/**
 * Perform a Newton method update step for a Newton iteration that determines
 * the local coordinates (xi) for a given set of physical coordinates. If
 * v0,v1,v2 and v3 (passed component wise) are the vertices of a linear sided
 * quadrilateral then this function performs the Newton update:
 * 
 * xi_{n+1} = xi_n - J^{-1}(xi_n) * F(xi_n)
 * 
 * where
 * 
 * F(xi) = X(xi) - X_phys
 * 
 * where X_phys are the global coordinates.
 * 
 * This is a generated function. To modify this function please edit the script
 * that generates this function.
 * 
 * @param[in] xi0 Current xi_n point, x component.
 * @param[in] xi1 Current xi_n point, y component.
 * @param[in] v00 Vertex 0, x component of quadrilateral.
 * @param[in] v01 Vertex 0, y component of quadrilateral.
 * @param[in] v10 Vertex 1, x component of quadrilateral.
 * @param[in] v11 Vertex 1, y component of quadrilateral.
 * @param[in] v20 Vertex 2, x component of quadrilateral.
 * @param[in] v21 Vertex 2, y component of quadrilateral.
 * @param[in] v30 Vertex 3, x component of quadrilateral.
 * @param[in] v31 Vertex 3, y component of quadrilateral.
 * @param[in] phys0 Target point in physical space, x component.
 * @param[in] phys1 Target point in physical space, y component.
 * @param[in, out] xin0 Output local coordinate iteration, x component.
 * @param[in, out] xin1 Output local coordinate iteration, y component.
 */
inline void newton_step_quad_linear(
const REAL xi0,
const REAL xi1,
const REAL v00,
const REAL v01,
const REAL v10,
const REAL v11,
const REAL v20,
const REAL v21,
const REAL v30,
const REAL v31,
const REAL phys0,
const REAL phys1,
REAL * xin0,
REAL * xin1
){"""
    )
    step_list = [step[0][xin0], step[0][xin1]]
    cse_list = cse(step_list, optimizations='basic')
    for cse_expr in cse_list[0]:
        lhs = cse_expr[0]
        rhs = cse_expr[1]
        expr = f"const REAL {lhs} = {rhs};"
        print(expr)

    expr_xin0 = f"const REAL xin0_tmp = {cse_list[1][0]};"
    expr_xin1 = f"const REAL xin1_tmp = {cse_list[1][1]};"
    print(expr_xin0)
    print(expr_xin1)
    print("*xin0 = xin0_tmp;")
    print("*xin1 = xin1_tmp;")
    print("}\n")



if __name__ == "__main__":

    x = get_x()
    f = x - phys
    J = f.jacobian(xi)

    step = solve(xin - xi + (J**(-1)) * f, xin, dict=True)
    step_list = [step[0][xin0], step[0][xin1]]


    vertices_ref = {
        v00 : -1.0,
        v01 : -1.0,
        v10 :  1.0,
        v11 : -1.0,
        v20 :  1.0,
        v21 :  1.0,
        v30 : -1.0,
        v31 :  1.0,
    }

    test_vertices = (
        (-1.0, -1.0),
        ( 1.0, -1.0),
        (-1.0,  1.0),
        ( 1.0,  1.0),
    )
    for vx in test_vertices:
        correct = np.array(vx).ravel()
        to_test = np.array(xi_to_phys(vertices_ref, vx[0], vx[1])).ravel()
        abs_error = np.linalg.norm(correct - to_test, np.inf)
        assert abs_error < 1.0E-14, f"self test failed at vertex {vx}, error {abs_error}"
    

    vertices = {
        v00 : -3.0,
        v01 : -2.0,
        v10 :  1.0,
        v11 : -1.0,
        v20 :  2.0,
        v21 :  2.0,
        v30 : -1.0,
        v31 :  4.0,
    }
    

    xi_correct0 = -0.9
    xi_correct1 = 0.8
    test_phys = xi_to_phys(vertices, xi_correct0, xi_correct1)
    test_phys0 = float(test_phys[0])
    test_phys1 = float(test_phys[1])

    xin0v = 0.0
    xin1v = 0.0

    res = newton_residual(vertices, xin0v, xin1v, test_phys0, test_phys1)
    for stepx in range(5):
        xin0v, xin1v = newton_step(vertices, xin0v, xin1v, test_phys0, test_phys1)
        res = newton_residual(vertices, xin0v, xin1v, test_phys0, test_phys1)

    assert abs(xi_correct0 - xin0v) < 1.0e-15, "self newton test failed"
    assert abs(xi_correct1 - xin1v) < 1.0e-15, "self newton test failed"

    print_x(step)







