from sympy import *
import sympy.printing.c
from sympy.codegen.rewriting import create_expand_pow_optimization


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

    def generate_variable(self, p):
        return symbols(f"modA_{p}_{self.z}")

    def __call__(self, p):
        assert p >= 0
        assert p < self.P
        v = self.generate_variable(p)
        return v

    def generate(self):
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


if __name__ == "__main__":

    eta0 = symbols("eta0")
    jacobi0 = GenJacobi(eta0)
    a = eModified_A(8, eta0, jacobi0)

    gen_mod0 = a.generate()
    gen_jacobi0 = jacobi0.generate()

    lines = gen_jacobi0 + gen_mod0


    s = ""
    for lx in lines:
        lhs = lx[0]
        rhs = sympy.printing.c.ccode(lx[1], standard="C99")
        s += f"const REAL {lhs} = {rhs};\n"

    print(s)
    
    steps = [lx[1] for lx in lines]
    instr = []
    cse_list = cse(steps, optimizations="basic")
    for cse_expr in cse_list[0]:
        lhs = cse_expr[0]
        e = sympy.printing.c.ccode(cse_expr[1], assign_to=lhs, standard="C99")
        expr = f"const REAL {e}"
        instr.append(expr)
    for lhs_v, rhs_v in zip([lx[0] for lx in lines], cse_list[1]):
        e = sympy.printing.c.ccode(rhs_v, assign_to=lhs_v, standard="C99")
        expr = f"const REAL {e}"
        instr.append(expr)   

    print("\n".join(instr))

