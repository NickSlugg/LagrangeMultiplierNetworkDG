import numpy as np
import numpy.typing as npt
from typing import List

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.plot
import ufl
import basix.ufl

import pyvista as pv

from fenicsx_ii import assemble_scalar

from mesh import GenMesh3D
from GraphProblem import GraphProblem

graph_prob = GraphProblem()

Nvals = [2, 4, 8, 16, 32]
results = np.zeros((len(Nvals), 3))

for i, N in enumerate(Nvals):
    results[i, 0] = N
    graph_prob.global_graph_mesh(N)
    graph_prob.graph_mesh_edges()
    graph_prob.create_function_spaces()

    x1D = ufl.SpatialCoordinate(graph_prob.graph_mesh)
    xlm = ufl.SpatialCoordinate(graph_prob.lm_mesh)

    uhat_1 = x1D[2] + ufl.cos(2 * ufl.pi * x1D[2])
    uhat_2 = 2 + 0.5 * ufl.sqrt(2) * (x1D[2] - 1)

    f_edges = [
        4 * (ufl.pi ** 2) * ufl.cos(2 * ufl.pi * x1D[2]),
        0,
        0,
    ]

    u_ex_edges = [
    uhat_1, 
    uhat_2, 
    uhat_2,
    2 * (xlm[0] ** 0),
    ]

    graph_prob.set_exact_soln(u_ex_edges, f_edges)
    graph_prob.setup_forms()
    graph_prob.solve_system()
    results[i, 1] = graph_prob.compute_error(u_ex_edges)
    graph_prob.reset_edges()

for i in range(1, len(Nvals)):
    results[i,2] = (np.log(results[i,1]) - np.log(results[i-1,1]))/np.log(0.5)

for i in range(results.shape[0]):
    line = ""
    for j in range(results.shape[1]-1):
        line += f" {results[i, j]} &"
    line += f" {results[i, 2]} \\\\ \n"
    print(line)


"""

u_ex_fun = []
for i, edge in enumerate(graph_prob.edge_spaces):
    u_ex_fun.append(dolfinx.fem.Function(edge))
    u_expr = dolfinx.fem.Expression(u_ex_edges[i], edge.element.interpolation_points)
    u_ex_fun[i].interpolate(u_expr)

plotter = pv.Plotter()

pv.global_theme.font.label_size = 30
pv.global_theme.font.title_size = 30

# Visualize the solution on each edge
#plotter.subplot(0,0)
for i, func in enumerate(graph_prob.functions[:-1]):  # Exclude the Lagrange multiplier
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(func.function_space)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["u"] = func.x.array.real
    grid.set_active_scalars("u")
    fun_plot = plotter.add_mesh(grid, line_width=10, cmap="viridis", show_scalar_bar=False)
    plotter.add_scalar_bar(title="", mapper=fun_plot.mapper)
plotter.view_xz()
plotter.show_bounds()

# Visualize error solution
plotter.subplot(0,1)
for i, (func, space) in enumerate(zip(graph_prob.functions[:-1], graph_prob.edge_spaces)):  # Exclude the Lagrange multiplier
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(func.function_space)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    err_exp = dolfinx.fem.Expression(u_ex_fun[i] - func, space.element.interpolation_points)
    err_fun = dolfinx.fem.Function(space)
    err_fun.interpolate(err_exp)
    grid.point_data["u"] = err_fun.x.array.real
    grid.set_active_scalars("u")
    err_plot = plotter.add_mesh(grid, line_width=10, cmap="viridis", show_scalar_bar=False)
    plotter.add_scalar_bar(title="Error", mapper=err_plot.mapper)

plotter.view_xz()
plotter.show_bounds()
plotter.show()
plotter.screenshot("Y-Bifurc-Error.png")
"""

