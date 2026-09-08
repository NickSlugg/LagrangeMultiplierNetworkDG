import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx.fem import assemble_scalar
import dolfinx.plot as plot
import ufl
from pathlib import Path

import networkx as nx
import networks_fenicsx
from networks_fenicsx import (
    NetworkMesh,
    HydraulicNetworkAssembler,
    network_generation
)
import pyvista

from assembly import PoissonAssembler
from solver import Solver
from post_processing import extract_graph_fun

if __name__ == "__main__":

    G = nx.DiGraph([(0, 1), (1, 2)])
    G.nodes[0]["pos"] = [0, 0]
    G.nodes[1]["pos"] = [0, 0.5]
    G.nodes[2]["pos"] = [0, 1]

    network_mesh = NetworkMesh(G, N=4)

    dx_graph = ufl.Measure("dx", domain=network_mesh.mesh)

    x = ufl.SpatialCoordinate(network_mesh.mesh)
    #u_bc_ex = ufl.conditional(x[1] < 0.5, ufl.sin(2 * ufl.pi * x[1]), 2 * x[1] - 1)
    #f_hat = ufl.conditional(x[1] < 0.5, 4 * (ufl.pi**2) *  ufl.sin(2 * ufl.pi * x[1]), 0)
    u_bc_ex = ufl.sin(x[1])
    f_hat = 

    assembler = PoissonAssembler(network_mesh)
    assembler.compute_forms(u_bc_ex=u_bc_ex, f_hat=f_hat)

    solver = Solver(assembler)
    solver.assemble()
    sol = solver.solve()

    global_sol = extract_graph_fun(network_mesh, sol)

    ## Plot the Solution
    cells, types, x = plot.vtk_mesh(global_sol.function_space)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = global_sol.x.array
    grid.set_active_scalars("u")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    plotter.show()

