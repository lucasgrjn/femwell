import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from skfem import Basis, ElementTriP0


def plot_subdomain_boundaries(mesh, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect(1)

    mesh.draw(ax=ax, boundaries=True, boundaries_only=True)
    for subdomain in mesh.subdomains.keys() - {"gmsh:bounding_entities"}:
        mesh.restrict(subdomain).draw(ax=ax, boundaries_only=True)
    return ax


def plot_domains(mesh, ax=None, cmap_dict=None):
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect(1)
    basis0 = Basis(mesh, ElementTriP0())

    subdomains = list(mesh.subdomains.keys() - {"gmsh:bounding_entities"})
    subdomain_colors = basis0.zeros() * np.NaN
    if cmap_dict is not None:
        plot_colors = []
        for i, subdomain in enumerate(subdomains):
            subdomain_colors[basis0.get_dofs(elements=subdomain)] = i / (len(subdomains) - 1)
            plot_colors.append(cmap_dict[subdomain])
        cmap = matplotlib.colors.ListedColormap(plot_colors)
        ax = basis0.plot(subdomain_colors, ax=ax, cmap=cmap)
        plt.colorbar(
            ax.collections[-1], ticks=[(ic + 0.5) / (i + 1) for ic in range(i + 1)]
        ).ax.set_yticklabels(subdomains)
    else:
        for i, subdomain in enumerate(subdomains):
            subdomain_colors[basis0.get_dofs(elements=subdomain)] = i
        norm = matplotlib.colors.BoundaryNorm(np.arange(i + 2) - 0.5, ncolors=256)
        ax = basis0.plot(subdomain_colors, plot_kwargs={"norm": norm}, ax=ax, cmap="rainbow")
        plt.colorbar(ax.collections[-1], ticks=list(range(i + 1))).ax.set_yticklabels(subdomains)

    return ax
