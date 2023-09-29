from sympy import *
import sympy.printing.c
from sympy.codegen.rewriting import create_expand_pow_optimization
import functools

class GenJacobi:
    def __init__(self, z, sym="P"):
        self.z = z
        self.sym = sym
        self.requested = set()

    def generate_variable(self, n, alpha, beta):
        return symbols(f"P_{n}_{alpha}_{beta}_{self.z}")

    def __call__(self, n, alpha, beta):
        self.requested.add((n, alpha, beta))
        v = self.generate_variable(n, alpha, beta)
        return v

    @staticmethod
    def _pochhammer(m, n):
        output = 1
        for offset in range(0, n):
            output *= m + offset
        return output

    def _generate_to_order(self, n_max, alpha, beta):

        g = []
        z = self.z

        for p in range(n_max + 1):
            s = self.generate_variable(p, alpha, beta)
            v = None

            if p == 0:
                v = Float(1.0)
            elif p == 1:
                v = 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (z - 1.0))
            elif p == 2:
                v = 0.125 * (
                    4 * (alpha + 1) * (alpha + 2)
                    + 4 * (alpha + beta + 3) * (alpha + 2) * (z - 1.0)
                    + (alpha + beta + 3)
                    * (alpha + beta + 4)
                    * (z - 1.0)
                    * (z - 1.0)
                )
            else:
                n = p - 1
                pn = self.generate_variable(n, alpha, beta)
                pnm1 = self.generate_variable(n - 1, alpha, beta)
                coeff_pnp1 = (
                    2
                    * (n + 1)
                    * (n + alpha + beta + 1)
                    * (2 * n + alpha + beta)
                )
                coeff_pn = (2 * n + alpha + beta + 1) * (
                    alpha * alpha - beta * beta
                ) + self._pochhammer(2 * n + alpha + beta, 3) * z
                coeff_pnm1 = (
                    -2.0 * (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2)
                )

                v = (1.0 / coeff_pnp1) * (coeff_pn * pn + coeff_pnm1 * pnm1)

            g.append((s, v))

        return g

    def generate(self):
        set_alpha_beta = set()
        for rx in self.requested:
            set_alpha_beta.add((rx[1], rx[2]))

        orders_alpha_beta = {}
        for abx in set_alpha_beta:
            orders_alpha_beta[abx] = 0
        for rx in self.requested:
            key = (rx[1], rx[2])
            current = orders_alpha_beta[key]
            orders_alpha_beta[key] = max(current, rx[0])

        g = []
        for abx in orders_alpha_beta.items():
            n_max = abx[1]
            alpha = abx[0][0]
            beta = abx[0][1]
            g += self._generate_to_order(n_max, alpha, beta)
        return g


class eModified_A:
    def __init__(self, P: int, z, jacobi):
        self.P = P
        self.z = z
        self.jacobi = jacobi
        self._g = self._generate()

    def generate_variable(self, p):
        return symbols(f"modA_{p}_{self.z}")

    def __call__(self, p):
        assert p >= 0
        assert p < self.P
        v = self.generate_variable(p)
        return v

    def _generate(self):
        g = []
        b0 = 0.5 * (1.0 - self.z)
        b1 = 0.5 * (1.0 + self.z)
        for p in range(self.P):
            s = self.generate_variable(p)
            if p == 0:
                g.append((s, b0))
            elif p == 1:
                g.append((s, b1))
            else:
                g.append((s, b0 * b1 * self.jacobi(p - 2, 1, 1)))
        return g

    def generate(self):
        return self._g


class PowOptimiser:
    _idx = 0

    def __init__(self, P, z):
        self.z = z
        self.P = P
        self._base_name = f"pow_base_{PowOptimiser._idx}"
        self._base = symbols(self._base_name)
        PowOptimiser._idx += 1
        self._g = self._generate()
    
    def generate_variable(self, p):
        assert p >= 0
        assert p <= self.P
        return symbols(f"pow_{p}_{self._base_name}")

    def generate(self):
        return self._g

    def _generate(self):
        g = [
            (self._base, self.z),
            (self.generate_variable(0), RealNumber(1.0))
        ]
        for px in range(1, self.P + 1):
            g.append(
                (self.generate_variable(px), self._base * self.generate_variable(px-1))
            )
        return g


class eModified_B:
    def __init__(self, P: int, z, jacobi):
        self.P = P
        self.z = z
        self.jacobi = jacobi
        self._modA = eModified_A(P, z, jacobi)
        b0 = 0.5 * (1.0 - self.z)
        self._pow = PowOptimiser(P-1, b0)
        self._g = self._modA.generate() + self._pow.generate() + self._generate()

    def generate_variable(self, p, q):
        return symbols(f"modB_{p}_{q}_{self.z}")

    def __call__(self, p, q):
        assert p >= 0
        assert p < self.P
        assert q >= 0
        assert q <= p

        v = self.generate_variable(p)
        return v

    def _generate(self):
        g = []
        b0 = 0.5 * (1.0 - self.z)
        b1 = 0.5 * (1.0 + self.z)
        for p in range(self.P):
            for q in range(self.P - p):
                s = self.generate_variable(p, q)
                if p == 0:
                    g.append((s, self._modA.generate_variable(q)))
                elif q == 0:
                    g.append((s, self._pow.generate_variable(p)))
                else:
                    g.append((s, self._pow.generate_variable(p) * b1 * self.jacobi(q - 1, 2 * p - 1, 1)))
        return g

    def generate(self):
        return self._g


def generate_statement(lhs, rhs, t):
    

    expand_pow = create_expand_pow_optimization(99)
    expr = sympy.printing.c.ccode(expand_pow(rhs), standard="C99")
    stmt = f"{t} {lhs}({expr});"
    return stmt


def generate_block(components, t="REAL"):
    g = []
    for cx in components:
        g.append(cx.generate())

    instr = []
    symbols_generator = numbered_symbols()

    ops = 0

    for gx in g:
        output_steps = [lx[0] for lx in gx]
        steps = [lx[1] for lx in gx]
        cse_list = cse(steps, symbols=symbols_generator, optimizations="basic")
        for cse_expr in cse_list[0]:
            lhs = cse_expr[0]
            ops += cse_expr[1].count_ops()
            e = generate_statement(lhs, cse_expr[1], t)
            instr.append(e)
        for lhs_v, rhs_v in zip(output_steps, cse_list[1]):
            e = generate_statement(lhs_v, rhs_v, t)
            ops += rhs_v.count_ops()
            instr.append(e)
    
    return instr, ops


class DofReader:
    def __init__(self, sym, max_index):
        self.sym = MatrixSymbol(sym, 1, max_index)
        self.max_index = max_index
        self._g = self._generate()

    def generate_variable(self, ix):
        return symbols(f"dof_{ix}")

    def generate(self):
        return self._g

    def _generate(self):
        g = []
        for ix in range(self.max_index):
            g.append((self.generate_variable(ix), self.sym[ix]))
        return g


class QuadrilateralEvaluate:
    def __init__(self, P, dofs, dir0, dir1):
        self.P = P
        self.dofs = dofs
        self.dir0 = dir0
        self.dir1 = dir1
        self.eta0 = dir0.z
        self.eta1 = dir1.z
        self._g = self._generate()

    def generate(self):
        return self._g

    def generate_variable(self):
        return symbols(f"eval_{self.eta0}_{self.eta1}")

    def _generate(self):
        ev = self.generate_variable()
        g = [
        ]
        
        mode = 0
        tmps = []
        for qx in range(self.P):
            for px in range(self.P):
                tmp_mode = symbols(f"eval_{self.eta0}_{self.eta1}_{mode}")
                tmps.append(tmp_mode)
                g.append(
                    (tmp_mode, self.dofs.generate_variable(mode) * self.dir0.generate_variable(px) * self.dir1.generate_variable(qx))
                )
                mode += 1

        g.append(
            (ev, functools.reduce(lambda x,y: x+y, tmps))
        )
        return g


def quadrilateral_evaluate_scalar(P, dofs, eta0, eta1):
    jacobi0 = GenJacobi(eta0)
    dir0 = eModified_A(P, eta0, jacobi0)
    jacobi1 = GenJacobi(eta1)
    dir1 = eModified_A(P, eta1, jacobi1)
    dof_reader = DofReader(dofs, P * P)
    loop = QuadrilateralEvaluate(P, dof_reader, dir0, dir1)
    

    blocks = [
        jacobi0, dir0,
        jacobi1, dir1,
        dof_reader,
        loop
    ]

    instr, ops = generate_block(blocks, "REAL")
    instr_str = "\n".join(["  " + ix for ix in instr])

    func = f"""
template <>
inline REAL quadrilateral_evaluate_scalar<{P}>(
  const REAL eta0,
  const REAL eta1,
  const NekDouble * dofs
){{
{instr_str}
  return {loop.generate_variable()};
}}
    """

    print(func)
    print(ops)

    return func
    



if __name__ == "__main__":

    eta0 = symbols("eta0")
    jacobi0 = GenJacobi(eta0)
    dir0 = eModified_A(8, eta0, jacobi0)
    instrA, opsA = generate_block((jacobi0, dir0))
    print("\n".join(instrA))
    
    print("-" * 60)
    eta1 = symbols("eta1")
    jacobi1 = GenJacobi(eta1)
    dir1 = eModified_B(8, eta1, jacobi1)
    instrB, opsB = generate_block((jacobi1, dir1))
    print("\n".join(instrB))

    print("-" * 60)
    dofs = IndexedBase("dofs")

    dofs = "dofs"
    quad = quadrilateral_evaluate_scalar(4, dofs, eta0, eta1)
    quad = quadrilateral_evaluate_scalar(8, dofs, eta0, eta1)
