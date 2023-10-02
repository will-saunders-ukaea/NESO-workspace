from sympy import *
from jacobi import *
from pow_optimiser import PowOptimiser


class eModified_A:
    def __init__(self, P: int, z, jacobi):
        self.P = P
        self.z = z
        self.jacobi = jacobi
        self._g = self._generate()

    def generate_variable(self, p):
        return symbols(f"modA_{p}_{self.z}")

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


class eModified_C:
    def __init__(self, P: int, z, jacobi):
        self.P = P
        self.z = z
        self._B = eModified_B(P, z, jacobi)
        self._g = self._B.generate() + self._generate()

    def generate_variable(self, p, q, r):
        return symbols(f"modC_{p}_{q}_{r}_{self.z}")

    def _generate(self):
        g = []
        for p in range(self.P):
            for q in range(self.P - p):
                for r in range(self.P - p - q):
                    g.append(
                        (
                            self.generate_variable(p,q,r), 
                            self._B.generate_variable(p+q, r)
                        )
                    )
        return g

    def generate(self):
        return self._g


class eModified_PyrC:
    def __init__(self, P: int, z, jacobi):
        self.P = P
        self.z = z
        self.jacobi = jacobi
        self._B = eModified_B(P, z, jacobi)
        b0 = 0.5 * (1.0 - self.z)
        self._pow = PowOptimiser(2*P-1, b0)
        self._g = self._B.generate() + self._generate()

    def generate_variable(self, p, q, r):
        return symbols(f"modPyrC_{p}_{q}_{r}_{self.z}")

    def _generate(self):
        g = []
        b0 = 0.5 * (1.0 - self.z)
        b1 = 0.5 * (1.0 + self.z)
        for p in range(self.P):
            for q in range(self.P):
                for r in range(self.P - max(p, q)):
                    lhs = self.generate_variable(p,q,r)
                    rhs = None
                    if p == 0:
                        rhs = self._B.generate_variable(q, r)
                    elif p == 1:
                        if q == 0:
                            rhs = self._B.generate_variable(1, r)
                        else:
                            rhs = self._B.generate_variable(q, r)
                    else:
                        if q < 2:
                            rhs = self._B.generate_variable(p, r)
                        else:
                            if r == 0:
                                rhs = self._pow.generate_variable(p + q - 2)
                            else:
                                rhs = self._pow.generate_variable(p + q - 2) * b1 * self.jacobi.generate_variable(r - 1, 2 * p + 2 * q - 3, 1)

                    assert rhs is not None
                    g.append(
                        (
                            lhs, 
                            rhs
                        )
                    )
        return g

    def generate(self):
        return self._g
