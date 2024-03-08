# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: env_femwell
#     language: python
#     name: python3
# ---

# +
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.affinity
from shapely.ops import clip_by_rect
from skfem import Basis, ElementDG, ElementTriP0, ElementTriP1, adaptive_theta
from skfem.io.meshio import from_meshio

from femwell.maxwell.waveguide import compute_modes, eval_error_estimator
from femwell.mesh import mesh_from_OrderedDict
from femwell.visualization import plot_domains, plot_subdomain_boundaries

# -

h_wg = 0.3
alpha = 60.0
w_wg = 1.0
h_TF = 0.6
h_INS = 2.0

# +
sim_window = [5 * w_wg, 4 * h_INS]

poly_tfln = shapely.Polygon(
    (
        (-sim_window[0] / 2, 0),
        (-sim_window[0] / 2, h_TF - h_wg),
        (-w_wg / 2 - h_wg / np.tan(np.deg2rad(alpha)), h_TF - h_wg),
        (-w_wg / 2, h_TF),
        (+w_wg / 2, h_TF),
        (+w_wg / 2 + h_wg / np.tan(np.deg2rad(alpha)), h_TF - h_wg),
        (+sim_window[0] / 2.0, h_TF - h_wg),
        (+sim_window[0] / 2.0, 0),
    )
)
poly_air = shapely.box(-sim_window[0] / 2, 0, +sim_window[0] / 2, +sim_window[1] / 2)
poly_ins = shapely.box(-sim_window[0] / 2, -h_INS, +sim_window[0] / 2, 0)
poly_sub = shapely.box(-sim_window[0] / 2, -sim_window[1] / 2, +sim_window[0] / 2, -h_INS)

clip_size = [-10, -3, 10, 3]
polygons = OrderedDict(
    si=clip_by_rect(poly_sub, *clip_size),
    tfln=clip_by_rect(poly_tfln, *clip_size),
    sio2=clip_by_rect(poly_ins, *clip_size),
    air=clip_by_rect(poly_air, *clip_size),
)

# +
resolutions = {
    "tfln": {"resolution": 0.05, "distance": 0.5},
    "air": {"resolution": 0.2, "distance": 0.5},
    "sio2": {"resolution": 0.2, "distance": 0.5},
    "si": {"resolution": 0.2, "distance": 0.5},
}

# mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions, default_resolution_max=0.05))
custom_colormap = {
    "sio2": np.array((189, 189, 189)) / 255,
    "air": np.array((189, 240, 255)) / 255,
    "si": np.array((176, 174, 120)) / 255,
    "tfln": np.array((0, 198, 255)) / 255,
}

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
mesh = from_meshio(mesh_from_OrderedDict(polygons, resolutions))

plot_domains(mesh, ax=axs[0], cmap_dict=custom_colormap)
axs[0].set_title("Structure considered")


mesh.draw(ax=axs[1])
axs[1].set_title("Mesh generated")
plot_subdomain_boundaries(mesh, ax=axs[1])

for ax in axs:
    ax.set_xlabel("x [$\\mu m$]")
    ax.set_ylabel("y [$\\mu m$]")
    ax.set_aspect("equal")

plt.show()

# +
fig, ax = plt.subplots()
ax.set_aspect(1)

basis0 = Basis(mesh, ElementTriP0())

subdomains = list(mesh.subdomains.keys() - {"gmsh:bounding_entities"})
subdomain_colors = basis0.zeros() * np.NaN
for i, subdomain in enumerate(subdomains):
    subdomain_colors[basis0.get_dofs(elements=subdomain)] = i

norm = matplotlib.colors.BoundaryNorm(np.arange(i + 2) - 0.5, ncolors=256)
ax = basis0.plot(subdomain_colors, plot_kwargs={"norm": norm}, ax=ax, cmap="inferno")
plt.colorbar(ax.collections[-1], ticks=list(range(i + 1))).ax.set_yticklabels(subdomains)
# -

basis0.plot()

# +
fig, ax = plt.subplots()
ax.set_aspect(1)

basis0 = Basis(mesh, ElementTriP0())

subdomains = list(mesh.subdomains.keys() - {"gmsh:bounding_entities"})
subdomain_colors = basis0.zeros() * np.NaN
for i, subdomain in enumerate(subdomains):
    subdomain_colors[basis0.get_dofs(elements=subdomain)] = i

ax = basis0.plot(subdomain_colors, plot_kwargs={"norm": norm}, ax=ax)
plt.colorbar(ax.collections[-1], ticks=list(range(i + 1))).ax.set_yticklabels(subdomains)

# +
import matplotlib.colors


def plot_domains2(mesh, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect(1)
    basis0 = Basis(mesh, ElementTriP0())

    subdomains = list(mesh.subdomains.keys() - {"gmsh:bounding_entities"})
    subdomain_colors = basis0.zeros() * np.NaN
    for i, subdomain in enumerate(subdomains):
        subdomain_colors[basis0.get_dofs(elements=subdomain)] = i

    norm = matplotlib.colors.BoundaryNorm(np.arange(i + 2) - 0.5, ncolors=256)
    ax = basis0.plot(subdomain_colors, plot_kwargs={"norm": norm}, ax=ax, cmap="rainbow")
    plt.colorbar(ax.collections[-1], ticks=list(range(i + 1))).ax.set_yticklabels(subdomains)

    return ax


plot_domains2(mesh)
# -

basis = Basis(mesh, ElementTriP1(), intorder=4)
basis_epsilon_r = basis.with_element(ElementTriP0())


g_h = 2.3
w_h = 3.1
L_h = 1.2 * 1e3
h_h = 0.1
