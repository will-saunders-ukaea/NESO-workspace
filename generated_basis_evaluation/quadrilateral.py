from sympy import *
import functools
from dof_reader import *

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



def quadrilateral_evaluate_vector(P, dofs, eta0, eta1):
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
    
    t = "sycl::vec<REAL, VECTOR_LENGTH>"
    instr, ops = generate_block(blocks, t)
    instr_str = "\n".join(["  " + ix for ix in instr])

    func = f"""
template <>
inline {t} quadrilateral_evaluate_vector<{P}>(
  const {t} eta0,
  const {t} eta1,
  const NekDouble * dofs
){{
{instr_str}
  return {loop.generate_variable()};
}}
    """

    print(func)
    print(ops)

    return func




