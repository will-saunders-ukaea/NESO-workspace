from sympy import *

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


