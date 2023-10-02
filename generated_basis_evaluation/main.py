from sympy import *
from basis_functions import *
from generate_source import *

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
    #quad = quadrilateral_evaluate_scalar(4, dofs, eta0, eta1)
    #quad = quadrilateral_evaluate_scalar(8, dofs, eta0, eta1)

    #quad = quadrilateral_evaluate_vector(4, dofs, eta0, eta1)
    #quad = quadrilateral_evaluate_vector(8, dofs, eta0, eta1)

    etaC = symbols("etaC")
    jacobiC = GenJacobi(etaC)
    dirC = eModified_C(8, etaC, jacobiC)
    instrC, opsC = generate_block((jacobiC, dirC))
    print("\n".join(instrC))


    etaPyrC = symbols("etaPyrC")
    jacobiPyrC = GenJacobi(etaPyrC)
    dirPyrC = eModified_PyrC(8, etaPyrC, jacobiPyrC)
    instrPyrC, opsPyrC = generate_block((jacobiPyrC, dirPyrC))
    print("\n".join(instrPyrC))

