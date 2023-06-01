from sympy import *
from sympy.codegen.rewriting import create_expand_pow_optimization
import numpy as np
init_printing(use_unicode=True)


def make_vector(*args):
    return Matrix([symbols(ax) for ax in args])

class SymbolicCommon:
    def __init__(self, geom):
        self.geom = geom
        self.ndim = geom.ndim
        self.xi = make_vector(*["xi{}".format(dx) for dx in range(self.ndim)])
        self.phys = make_vector(*["phys{}".format(dx) for dx in range(self.ndim)])
        self._x = self.geom.get_x(self.xi)
        self._f = self.geom.get_f(self.xi, self.phys)


class NewtonCommon(SymbolicCommon):
    def __init__(self, geom):
        SymbolicCommon.__init__(self, geom)
        self.xi_next = make_vector(*["xin{}".format(dx) for dx in range(self.ndim)])
        self.fv = make_vector(*["f{}".format(dx) for dx in range(self.ndim)])

class LinearEvaluateCommon:
    def __init__(self, geom, vertices):
        assert len(vertices) == geom.num_vertices
        self.vertices = vertices
        self.sub_vertices = {}
        for vx in range(geom.num_vertices):
            for dimx in range(self.ndim):
                self.sub_vertices[geom.vertices[vx][dimx]] = self.vertices[vx][dimx]


class Newton(NewtonCommon):
    def __init__(self, geom):
        NewtonCommon.__init__(self, geom)
        self.J = self._f.jacobian(self.xi)
        self.step = solve(self.xi_next - self.xi + (self.J**(-1)) * self.fv, self.xi_next, dict=True)
        self.step_components = [self.step[0][self.xi_next[dimx]] for dimx in range(self.ndim)]
        self.f = self._f
        self.x = self._x


class NewtonEvaluate:
    def __init__(self, newton, evaluate):
        self.newton = newton
        self.evaluate = evaluate
        assert self.newton.geom == evaluate.geom

    def residual(self, xi, phys):
        subs = {}
        subs.update(self.evaluate.sub_vertices)
        for dimx in range(self.evaluate.geom.ndim):
            subs[self.evaluate.phys[dimx]] = phys[dimx]
            subs[self.evaluate.xi[dimx]] = xi[dimx]

        e = self.newton.f.evalf(subs=subs)
        efloat = [float(ex) for ex in e]

        r = 0.0
        for ex in efloat:
            r = max(r, abs(ex))
        return r, efloat


    def step(self, xi, phys, fv):
        subs = {}
        subs.update(self.evaluate.sub_vertices)
        ndim = self.evaluate.geom.ndim
        for dimx in range(ndim):
            subs[self.evaluate.phys[dimx]] = phys[dimx]
            subs[self.evaluate.xi[dimx]] = xi[dimx]
            subs[self.newton.fv[dimx]] = fv[dimx]

        xin = [self.newton.step_components[dimx].evalf(subs=subs) for dimx in range(ndim)]
        xinfloat = [float(ex) for ex in xin]
        return xinfloat


class NewtonLinearCCode:
    def __init__(self, newton):
        self.newton = newton
        

    def residual(self):
        ndim = self.newton.geom.ndim
        expand_opt = create_expand_pow_optimization(99)


        component_name = ("x", "y", "z")
        docs_params = []
        args = []
        for dimx in range(ndim):
            n = f"xi{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(f"@param[in] {n} Current xi_n point, {component_name[dimx]} component.")

        for vi, vx in enumerate(self.newton.geom.vertices):
            for dimx in range(ndim):
                n = f"v{vi}{dimx}"
                args.append(
                    f"const REAL {n}"
                )
                docs_params.append(f"@param[in] {n} Vertex {vi}, {component_name[dimx]} component.")

        for dimx in range(ndim):
            n = f"phys{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(f"@param[in] {n} Target point in global space, {component_name[dimx]} component.")

        for dimx in range(ndim):
            n = f"f{dimx}"
            args.append(f"REAL * {n}")
            docs_params.append(f"@param[in, out] {n} Current f evaluation at xi, {component_name[dimx]} component.")

        params = "\n * ".join(docs_params)
        x_description = "\n * ".join(self.newton.geom.x_description.split("\n"))
        
        docstring = f"""
/**
 * Compute and return F evaluation where
 * 
 * F(xi) = X(xi) - X_phys
 * 
 * where X_phys are the global coordinates. X is defined as
 * 
 * {x_description}
 *
 * This is a generated function. To modify this function please edit the script
 * that generates this function.
 *
 * {params}
 */
"""
        
        args_string = ",\n".join(args)
        name = self.newton.geom.name
        s = f"""inline void newton_f_{name}(
{args_string}
)"""
        
        instr = ["{"]
        steps = [fx for fx in self.newton.f]
        cse_list = cse(steps, optimizations='basic')
        for cse_expr in cse_list[0]:
            lhs = cse_expr[0]
            rhs = expand_opt(cse_expr[1])
            expr = f"const REAL {lhs} = {rhs};"
            instr.append(expr)

        for dimx in range(ndim):
            instr.append(
                f"const REAL f{dimx}_tmp = {cse_list[1][dimx]};"
            )
        for dimx in range(ndim):
            instr.append(
                f"*f{dimx} = f{dimx}_tmp;"
            )

        s+="\n  ".join(instr)
        s += "\n}\n"

        return docstring + s


    def step(self):
        ndim = self.newton.geom.ndim
        expand_opt = create_expand_pow_optimization(99)


        component_name = ("x", "y", "z")
        docs_params = []
        args = []
        for dimx in range(ndim):
            n = f"xi{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(f"@param[in] {n} Current xi_n point, {component_name[dimx]} component.")

        for vi, vx in enumerate(self.newton.geom.vertices):
            for dimx in range(ndim):
                n = f"v{vi}{dimx}"
                args.append(
                    f"const REAL {n}"
                )
                docs_params.append(f"@param[in] {n} Vertex {vi}, {component_name[dimx]} component.")

        for dimx in range(ndim):
            n = f"phys{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(f"@param[in] {n} Target point in global space, {component_name[dimx]} component.")

        for dimx in range(ndim):
            n = f"f{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(f"@param[in] {n} Current f evaluation at xi, {component_name[dimx]} component.")

        for dimx in range(ndim):
            n = f"xin{dimx}"
            args.append(f"REAL * {n}")
            docs_params.append(f"@param[in, out] {n} Output local coordinate iteration, {component_name[dimx]} component.")

        params = "\n * ".join(docs_params)
        x_description = "\n * ".join(self.newton.geom.x_description.split("\n"))
        
        docstring = f"""
/**
 * Perform a Newton method update step for a Newton iteration that determines
 * the local coordinates (xi) for a given set of physical coordinates. If
 * v0,v1,v2 and v3 (passed component wise) are the vertices of a linear sided
 * quadrilateral then this function performs the Newton update:
 * 
 * xi_{{n+1}} = xi_n - J^{{-1}}(xi_n) * F(xi_n)
 * 
 * where
 * 
 * F(xi) = X(xi) - X_phys
 * 
 * where X_phys are the global coordinates. 
 * 
 * X is defined as
 * 
 * {x_description}
 *
 * This is a generated function. To modify this function please edit the script
 * that generates this function.
 *
 * {params}
 */
"""
        
        args_string = ",\n".join(args)
        name = self.newton.geom.name
        s = f"""inline void newton_step_{name}(
{args_string}
)"""
        
        instr = ["{"]
        cse_list = cse(self.newton.step_components, optimizations='basic')
        for cse_expr in cse_list[0]:
            lhs = cse_expr[0]
            rhs = expand_opt(cse_expr[1])
            expr = f"const REAL {lhs} = {rhs};"
            instr.append(expr)

        for dimx in range(ndim):
            instr.append(
                f"const REAL xin{dimx}_tmp = {cse_list[1][dimx]};"
            )
        for dimx in range(ndim):
            instr.append(
                f"*xin{dimx} = xin{dimx}_tmp;"
            )
        s+="\n  ".join(instr)
        s += "\n}\n"

        return docstring + s



class LinearGeomEvaluate(SymbolicCommon):
    def __init__(self, geom, vertices):
        SymbolicCommon.__init__(self, geom)
        LinearEvaluateCommon.__init__(self, geom, vertices)


    def x(self, xi):
        subs = {}
        for dimx in range(self.ndim):
            subs[self.xi[dimx]] = xi[dimx]
        subs.update(self.sub_vertices)
        return [float(fx) for fx in self._x.evalf(subs=subs)]


    def f(self, xi, phys):
        subs = {}
        for dimx in range(self.ndim):
            subs[self.xi[dimx]] = xi[dimx]
            subs[self.phys[dimx]] = phys[dimx]
        subs.update(self.sub_vertices)
        return [float(fx) for fx in self._f.evalf(subs=subs)]


class LinearQuad:
    def __init__(self):

        self.num_vertices = 4
        self.ndim = 2
        self.vertices = [make_vector("v{}0".format(vx), "v{}1".format(vx)) for vx in range(self.num_vertices)]
        self.name = "linear_2d"
        self.x_description = """
X(xi) = 0.25 * v0 * (1 - xi_0) * (1 - xi_1) + 
        0.25 * v1 * (1 + xi_0) * (1 - xi_1) + 
        0.25 * v3 * (1 - xi_0) * (1 + xi_1) + 
        0.25 * v2 * (1 + xi_0) * (1 + xi_1)
"""

    def get_x(self, xi):
        
        v = self.vertices
        x = Matrix(
            [
                0.25 * v[0][0] * (1 - xi[0]) * (1 - xi[1]) + \
                0.25 * v[1][0] * (1 + xi[0]) * (1 - xi[1]) + \
                0.25 * v[3][0] * (1 - xi[0]) * (1 + xi[1]) + \
                0.25 * v[2][0] * (1 + xi[0]) * (1 + xi[1]),
                0.25 * v[0][1] * (1 - xi[0]) * (1 - xi[1]) + \
                0.25 * v[1][1] * (1 + xi[0]) * (1 - xi[1]) + \
                0.25 * v[3][1] * (1 - xi[0]) * (1 + xi[1]) + \
                0.25 * v[2][1] * (1 + xi[0]) * (1 + xi[1])
            ]
        )
        return x

    def get_f(self, xi, phys):
        x = self.get_x(xi)
        f = x - phys
        return f




if __name__ == "__main__":

    quad = LinearQuad()
    
    vertices_ref = (
        (-1.0, -1.0),
        ( 1.0, -1.0),
        ( 1.0, 1.0),
        (-1.0, 1.0),
    )
    quad_ref = LinearGeomEvaluate(quad, vertices_ref)
    
    
    
    for vx in vertices_ref:
        to_test = quad_ref.x(vx)
        correct = vx
        assert np.linalg.norm(np.array(correct).ravel() - np.array(to_test).ravel(), np.inf) < 1.0E-15
        to_test = quad_ref.f(vx, vx)
        assert np.linalg.norm(np.array(to_test).ravel(), np.inf) < 1.0E-15
    
    
    vertices_test = (
        (-3.0, -2.0),
        ( 1.0, -1.0),
        ( 2.0, 2.0),
        (-1.0, 4.0),
    )
    quad_test = LinearGeomEvaluate(quad, vertices_test)
    quad_newton = Newton(quad)
    
    quad_newton_ccode = NewtonLinearCCode(quad_newton)
    
    print(quad_newton_ccode.residual())
    print(quad_newton_ccode.step())
    
    
    quad_newton_evaluate = NewtonEvaluate(quad_newton, quad_test)
    
    
    xi_correct0 = -0.9
    xi_correct1 = 0.8
    xi_correct = (xi_correct0, xi_correct1)
    phys = quad_test.x(xi_correct)
    residual, fv = quad_newton_evaluate.residual(xi_correct, phys)
    assert residual < 1.0E-15
    
    
    xi = [0.0, 0.0]
    for stepx in range(5):
        residual, fv = quad_newton_evaluate.residual(xi, phys)
        xin = quad_newton_evaluate.step(xi, phys, fv)
        xi[0] = xin[0]
        xi[1] = xin[1]
    
    assert abs(xi[0] - xi_correct[0]) < 1.0E-14
    assert abs(xi[1] - xi_correct[1]) < 1.0E-14
    
    
