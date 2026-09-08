import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.plot
import ufl
import basix.ufl
from dolfinx.fem.petsc import LinearProblem

from fenicsx_ii import assemble_scalar

from mesh import GenMesh3D

def Eval1DGraph(N):
    comm = MPI.COMM_WORLD
    _, _, graph_nodes, graph_cells = GenMesh3D(2*N, discontinuous=False)

    ufl_int = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(3,)))

    graph_mesh = dolfinx.mesh.create_mesh(
        comm,
        e=ufl_int,
        cells=graph_cells,
        x=graph_nodes,
    )

    degree = 1
    V = dolfinx.fem.functionspace(graph_mesh, ("CG", degree))
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)

    dx = ufl.Measure("dx", domain=graph_mesh)
    dS = ufl.Measure("dS", domain=graph_mesh)
    ds = ufl.Measure("ds", domain=graph_mesh)

    n_graph = ufl.FacetNormal(graph_mesh)
    h_graph = ufl.CellDiameter(graph_mesh)

    # Bilinear Form
    a = ufl.inner(ufl.grad(u), ufl.grad(w)) * dx

    # Manufactured Solution
    x = ufl.SpatialCoordinate(graph_mesh)

    uhat_1 = x[2] + ufl.cos(2 * ufl.pi * x[2])
    uhat_2 = 2 + 0.5 * ufl.sqrt(2) * (x[2] - 1)
    u_ex = ufl.conditional(x[2] < 1, uhat_1, uhat_2)

    # BC Expr
    graph_mesh.topology.create_connectivity(graph_mesh.topology.dim - 1, graph_mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(graph_mesh.topology)
    exterior_dofs = dolfinx.fem.locate_dofs_topological(
        V, graph_mesh.topology.dim - 1, exterior_facets
    )

    bc_expr = dolfinx.fem.Expression(u_ex, V.element.interpolation_points)
    u_bc = dolfinx.fem.Function(V)
    u_bc.interpolate(bc_expr)
    bc = dolfinx.fem.dirichletbc(u_bc, exterior_dofs)

    # RHS Function
    f = ufl.conditional(x[2] < 1, 4 * (ufl.pi ** 2) * ufl.cos(2 * ufl.pi * x[2]), 0)

    # Linear Form
    L = ufl.inner(f, w) * dx
    #L += ufl.inner(u_bc * n_graph, ufl.grad(w)) * ds

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="Poisson",
    )

    uh = problem.solve()
    return uh, u_ex, dx

def L2(f: ufl.core.expr.Expr, dx: ufl.Measure) -> float:
    integral = ufl.inner(f, f) * dx
    return np.sqrt(assemble_scalar(integral, op=MPI.SUM))

def arr2tab(array: np.typing.NDArray):
    tab = ""
    for i in range(array.shape[0]):
        for j in range(array.shape[1]-1):
            tab += f" {array[i, j]} &"
        tab += f" {array[i, j+1]} \\\\ \n"
    return tab

if __name__ == "__main__":

    results = np.zeros((5, 3))

    for i in range(5):
        uh, u_ex, dx = Eval1DGraph(2**(i+1))
        results[i,0] = 2**(i+1)
        results[i,1] = L2(uh - u_ex, dx)
        if i > 0:
            results[i,2] = (np.log(results[i,1]) - np.log(results[i-1,1]))/(np.log(0.5))

    print(arr2tab(results))
