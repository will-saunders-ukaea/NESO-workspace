from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import math
from firedrake.pyplot import triplot
from firedrake.__future__ import interpolate


def triangle_cartesian_to_barycentric(triangle, x, y):
    x1 = triangle[0, 0]
    y1 = triangle[0, 1]
    x2 = triangle[1, 0]
    y2 = triangle[1, 1]
    x3 = triangle[2, 0]
    y3 = triangle[2, 1]
    scaling = 1.0 / (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    l1 = scaling * ((x2 * y3 - x3 * y2) + (y2 - y3) * x + (x3 - x2) * y)
    l2 = scaling * ((x3 * y1 - x1 * y3) + (y3 - y1) * x + (x1 - x3) * y)
    l3 = scaling * ((x1 * y2 - x2 * y1) + (y1 - y2) * x + (x2 - x1) * y)
    return np.array((l1, l2, l3))


def quad_cartesian_to_collapsed(quad, x, y):
    x0 = quad[0, 0]
    y0 = quad[0, 1]
    x1 = quad[1, 0]
    y1 = quad[1, 1]
    x2 = quad[2, 0]
    y2 = quad[2, 1]
    x3 = quad[3, 0]
    y3 = quad[3, 1]
    a0 = 0.25 * (x0 + x1 + x2 + x3)
    a1 = 0.25 * (-x0 + x1 + x2 - x3)
    a2 = 0.25 * (-x0 - x1 + x2 + x3)
    a3 = 0.25 * (x0 - x1 + x2 - x3)
    b0 = 0.25 * (y0 + y1 + y2 + y3)
    b1 = 0.25 * (-y0 + y1 + y2 - y3)
    b2 = 0.25 * (-y0 - y1 + y2 + y3)
    b3 = 0.25 * (y0 - y1 + y2 - y3)
    A = a1 * b3 - a3 * b1
    B = -x * b3 + a0 * b3 + a1 * b2 - a2 * b1 + a3 * y - a3 * b0
    C = -x * b2 + a0 * b2 + a2 * y - a2 * b0
    determinate_inner = B * B - 4.0 * A * C
    determinate = math.sqrt(determinate_inner) if determinate_inner > 0 else 0.0
    i2A = 1.0 / (2.0 * A)
    Bpos = B >= 0.0
    eta0p = ((-B - determinate) * i2A) if Bpos else (2 * C) / (-B + determinate)
    eta0m = ((2.0 * C) / (-B - determinate)) if Bpos else (-B + determinate) * i2A
    eta1p = (y - b0 - b1 * eta0p) / (b2 + b3 * eta0p)
    eta1m = (y - b0 - b1 * eta0m) / (b2 + b3 * eta0m)

    if (abs(eta0p) <= 1.0) and (abs(eta1p) <= 1.0):
        return np.array((eta0p, eta1p))
    if (abs(eta0m) <= 1.0) and (abs(eta1m) <= 1.0):
        return np.array((eta0m, eta1m))
    return None


def quad_collapsed_to_barycentric(eta):
    etat0 = eta[0]
    etat1 = eta[1]
    xi0 = (etat0 + 1.0) * 0.5
    xi1 = (etat1 + 1.0) * 0.5
    l0 = (1.0 - xi0) * (1.0 - xi1)
    l1 = xi0 * (1.0 - xi1)
    l2 = xi0 * xi1
    l3 = (1.0 - xi0) * xi1
    return np.array((l0, l1, l2, l3))


def triangle():
    triangle = np.array(((0.0, 0.0), (2.0, 0.0), (0.0, 4.0)))
    values = np.array((1.0, 2.0, 3.0))
    N = 1000
    x = np.random.uniform(-0.1, 5.0, N)
    y = np.random.uniform(-0.1, 5.0, N)
    xx = []
    yy = []
    zz = []
    for ix in range(N):
        l = triangle_cartesian_to_barycentric(triangle, x[ix], y[ix])
        if np.sum(np.abs(l)) <= 1.0:
            xx.append(x[ix])
            yy.append(y[ix])
            zz.append(np.dot(l, values))
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_trisurf(xx, yy, zz, linewidth=0.2, antialiased=True)
    plt.show()


def quad():
    quad = np.array(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (1.0, 3.0),
            (0.0, 4.0),
        )
    )
    values = np.array((1.0, 2.0, 3.0, 4.0))

    mesh = RectangleMesh(1, 1, 1.0, 1.0, quadrilateral=True)

    mesh.coordinates.dat.data[0, :] = quad[0, :]
    mesh.coordinates.dat.data[1, :] = quad[3, :]
    mesh.coordinates.dat.data[2, :] = quad[2, :]
    mesh.coordinates.dat.data[3, :] = quad[1, :]

    print(mesh.coordinates.dat.data[:, :])

    V = FunctionSpace(mesh, "DQ", 1)
    u = Function(V)
    W = VectorFunctionSpace(mesh, V.ufl_element())
    X = assemble(interpolate(mesh.coordinates, W))

    N = X.dat.data_ro.shape[0]
    for ix in range(N):
        xx = X.dat.data_ro[ix, 0]
        yy = X.dat.data_ro[ix, 1]
        eta = quad_cartesian_to_collapsed(quad, xx, yy)
        l = quad_collapsed_to_barycentric(eta)
        u.dat.data[ix] = np.dot(l, values)

    N = 10000
    x = np.random.uniform(-1.0, 5.0, N)
    y = np.random.uniform(-1.0, 5.0, N)
    xx = []
    yy = []
    zz = []
    zz2 = []
    for ix in range(N):
        eta = quad_cartesian_to_collapsed(quad, x[ix], y[ix])
        if eta is not None:
            l = quad_collapsed_to_barycentric(eta)
            xx.append(x[ix])
            yy.append(y[ix])
            ff = np.dot(l, values)
            gg = u.at(x[ix], y[ix])
            zz.append(ff)
            zz2.append(gg)
            assert abs(ff - gg) < 1.0e-14

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot_trisurf(xx, yy, zz, linewidth=0.2, antialiased=True)
    ax.plot_trisurf(xx, yy, zz2, linewidth=0.2, antialiased=True)
    plt.show()


triangle()
quad()
