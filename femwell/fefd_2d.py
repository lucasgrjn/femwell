from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import shapely
from skfem import *
from skfem.helpers import grad, inner
from skfem.io import from_meshio

from femwell.mesh import mesh_from_OrderedDict

out = shapely.LineString(((6, 0), (6, 1)))
waveguide = shapely.box(0, 0, 10, 1)
mesh = from_meshio(
    mesh_from_OrderedDict(
        OrderedDict(out=out, waveguide=waveguide),
        resolutions={},
        default_resolution_max=0.02,
        filename="test.msh",
    )
)

basis = Basis(mesh, ElementTriP1())
basis0 = basis.with_element(ElementTriP0())
epsilon_r = basis0.zeros(dtype=complex) + 1
dofs = basis0.get_dofs(elements=lambda x: (x[0] > 4.48))  # *(x[1]>.5))
epsilon_r[dofs] = 2**2
# dofs = basis0.get_dofs(elements=lambda x: (x[0] > 6))  # *(x[1]>.5))
# epsilon_r[dofs] = 1**2
dofs = basis0.get_dofs(elements=lambda x: x[0] > 7)
epsilon_r[dofs] += basis0.project(lambda x: np.maximum(0, x[0] - 7) ** 2 * 0.5j, dtype=complex)[
    dofs
]
basis0.plot(epsilon_r.imag, shading="gouraud", colorbar=True)
plt.show()
input_basis = basis.boundary(lambda x: x[0] == np.min(x[0]))
output_basis = basis.boundary("out")

mu_r = 1
k0 = 6


def h_m(y, b, m):
    return np.sqrt((1 if m == 0 else 2) / b) * np.sin(m * np.pi * y / b)


def gamma_m(b, m):
    return np.sqrt((m * np.pi / b) ** 2 - k0**2, dtype=complex)


@BilinearForm(dtype=complex)
def maxwell(u, v, w):
    return 0.5 * (1 / w.epsilon_r * inner(grad(u), grad(v))) - k0**2 * inner(u, v)


@BilinearForm(dtype=complex)
def input_form(u, v, w):
    return 0.5 * inner(
        u,
        np.sum([h_m(w.x[1], 1, m) * gamma_m(1, m) * inner(h_m(w.x[1], 1, m), v) for m in range(3)]),
    )


@LinearForm(dtype=complex)
def input_form_rhs(v, w):
    return -2 * gamma_m(1, m=1) * inner(h_m(w.x[1], 1, m=1), v)


I = input_form.assemble(input_basis)
Ir = input_form_rhs.assemble(input_basis)

A = maxwell.assemble(basis, epsilon_r=basis0.interpolate(epsilon_r))
B = basis.zeros(dtype=complex)
input_dofs = basis.get_dofs(lambda x: x[0] == np.min(x[0]))
# B[input_dofs] = basis.project(lambda x: h_m(x[1], 1, 1), dtype=complex)[input_dofs]

C = solve(
    *condense(
        A + I,
        Ir,
        x=B,
        D=basis.get_dofs(
            {
                # lambda x: x[0] == np.min(x[0]),
                lambda x: x[1] == np.min(x[1]),
                lambda x: x[1] == np.max(x[1]),
            }
        ),
        expand=True,
    )
)
# C = solve(A,B)
C_abs = basis.project(np.abs(basis.interpolate(C)) ** 2)
basis.plot(C_abs, shading="gouraud", colorbar=True)
plt.show()


@Functional(dtype=complex)
def coefficient(w):
    return -inner(h_m(w.x[1], 1, m=w.m), w.H)


print(np.abs(coefficient.assemble(input_basis, H=input_basis.interpolate(C), m=1)))
print(np.abs(coefficient.assemble(output_basis, H=output_basis.interpolate(C), m=1)))
