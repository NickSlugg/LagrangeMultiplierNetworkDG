import numpy as np
import numpy.typing as npt
from typing import List
import matplotlib.pyplot as plt

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

Nvals = [2, 4, 8, 16, 32, 64, 128]
results = np.zeros((len(Nvals), 4))

for i, N in enumerate(Nvals):
    results[i, 0] = N
    graph_prob.global_graph_mesh(N)
    graph_prob.graph_mesh_edges()
    graph_prob.create_function_spaces()

    x1D = ufl.SpatialCoordinate(graph_prob.graph_mesh)
    xlm = ufl.SpatialCoordinate(graph_prob.lm_mesh)

    uhat_1 = x1D[2] + ufl.cos(2 * ufl.pi * x1D[2])
    #uhat_1 = x1D[2] + 1
    uhat_2 = 2 + 0.5 * ufl.sqrt(2) * (x1D[2] - 1)
    #uhat_2 = 2 + 0.5 * ufl.sqrt(2) * ufl.sin(x1D[2] - 1)

    f_edges = [
        4 * ufl.pi ** 2 * ufl.cos(2 * ufl.pi * x1D[2]),
        0, 
        0
    ]

    u_ex_edges = [
    uhat_1, 
    uhat_2, 
    uhat_2,
    2 * (xlm[0] ** 0),
    ]

    u_ex_fun = []
    for j in range(3):
        u_ex_fun.append(dolfinx.fem.Function(graph_prob.edge_spaces[j]))
        u_ex_fun[-1].interpolate(dolfinx.fem.Expression(u_ex_edges[j], graph_prob.edge_spaces[j].element.interpolation_points))

    graph_prob.set_exact_soln(u_ex_edges, f_edges)
    graph_prob.setup_forms()
    graph_prob.solve_system()
    result = graph_prob.compute_error(u_ex_fun)[0]
    results[i, 1], results[i, 3] = graph_prob.compute_error(u_ex_fun)
    graph_prob.reset_edges()


for i in range(1, len(Nvals)):
    results[i,2] = (np.log(results[i,1]) - np.log(results[i-1,1]))/np.log(0.5)

for i in range(results.shape[0]):
          print(f"{int(results[i, 0])} & {results[i,1]:.2e} & {results[i,2]:.2e} & {results[i, 3]:.2e} \\\\")

print(graph_prob.functions[-1].x.array)

"""
for i in range(3):
    V = graph_prob.edge_spaces[i]
    u = graph_prob.functions[i]
    edge = graph_prob.edge_meshes[i]

    dx_edge = ufl.Measure("dx", domain=edge)

    err_int = ufl.inner((u - u_ex_edges[i]), (u - u_ex_edges[i])) * dx_edge

    print(f"Edge {i}: {np.sqrt(assemble_scalar(err_int, op=MPI.SUM))}")


V = graph_prob.edge_spaces[1]
u = dolfinx.fem.Function(V)
u.interpolate(dolfinx.fem.Expression(ufl.grad(graph_prob.functions[1]), V.element.interpolation_points))
u_ex = dolfinx.fem.Function(V)
u_ex.interpolate(dolfinx.fem.Expression(ufl.grad(u_ex_edges[1]), V.element.interpolation_points))

u_ex_np = lambda x: 2 + 0.5 * np.sqrt(2) * (x - 1)

dof_coords = V.tabulate_dof_coordinates()[:,2]

dofmap = V.dofmap.list

num_cells = graph_prob.edge_meshes[1].topology.index_map(1).size_local

plt.figure(figsize=(8, 5))
first_segment = True

for cell in range(num_cells):
    cell_dofs = dofmap[cell]

    x_cell = dof_coords[cell_dofs]
    u_cell = u.x.array[cell_dofs]
    u_ex_cell = u_ex.x.array[cell_dofs]

    sort_idx = np.argsort(x_cell)

    label = "DG Solution" if first_segment else ""
    plt.plot(x_cell[sort_idx], u_cell[sort_idx], color="blue", linewidth=2, label=label)
    plt.plot(x_cell[sort_idx], u_ex_cell[sort_idx], color="red", linewidth=2)

    first_segment = False

plt.title("DG soln on line 1")
plt.xlabel('y')
plt.ylabel("u")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
"""

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

