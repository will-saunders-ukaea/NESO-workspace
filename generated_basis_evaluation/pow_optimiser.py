from sympy import *

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


