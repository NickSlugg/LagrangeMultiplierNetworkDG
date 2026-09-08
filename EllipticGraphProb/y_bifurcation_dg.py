import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.plot
import ufl
import basix.ufl
from dolfinx.fem.petsc import LinearProblem

from fenicsx_ii import assemble_scalar

from mesh import GenMesh3D

def Eval1DGraph(N, plotting=False):
    comm = MPI.COMM_WORLD
    _, _, graph_nodes, graph_cells = GenMesh3D(2*N, discontinuous=False)

    ufl_int = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(3,)))

    graph_mesh = dolfinx.mesh.create_mesh(
        comm,
        e=ufl_int,
        cells=graph_cells,
        x=graph_nodes,
    )

    lm_mesh = dolfinx.mesh.create_submesh(
        graph_mesh,
        0,
        np.array([N], dtype=np.int32)
    )

    degree = 1
    V = dolfinx.fem.functionspace(graph_mesh, ("DG", degree))
    Vtilde = dolfinx.fem.functionspace(graph_mesh, ("DG", 0))
    W = ufl.MixedFunctionSpace(*[V, Vtilde])

    u, utilde = ufl.TrialFunctions(W)
    w, wtilde = ufl.TestFunctions(W)

    tdim = graph_mesh.topology.dim
    vertex_map = graph_mesh.topology.index_map(tdim-1)
    num_vertices = vertex_map.size_local + vertex_map.num_ghosts
    marker = np.ones(num_vertices, dtype=np.int32)
    marker[N] = 2
    vertex_tag = dolfinx.mesh.meshtags(graph_mesh, tdim-1, np.arange(num_vertices), marker)

    dx = ufl.Measure("dx", domain=graph_mesh)
    dS = ufl.Measure("dS", domain=graph_mesh, subdomain_data=vertex_tag)
    ds = ufl.Measure("ds", domain=graph_mesh)

    n_graph = ufl.FacetNormal(graph_mesh)
    h_graph = ufl.CellDiameter(graph_mesh)

    sigma_e = 30
    sigma_v = 30

    # Bilinear Form
    a = ufl.inner(ufl.grad(u), ufl.grad(w)) * dx
    a -= ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(w, n_graph)) * dS
    a -= ufl.inner(ufl.avg(ufl.grad(w)), ufl.jump(u, n_graph)) * dS
    a += (sigma_e / ufl.avg(h_graph)) * ufl.inner(ufl.jump(u, n_graph), ufl.jump(w, n_graph)) * dS
    a -= ufl.inner(ufl.grad(u), ufl.outer(w, n_graph)) * ds
    a -= ufl.inner(ufl.grad(w), ufl.outer(u, n_graph)) * ds
    a += (sigma_e / h_graph) * ufl.inner(u, w) * ds

    # Bifurcation Node Terms
    a += ufl.inner(ufl.grad(u), ufl.jump(w - wtilde, n_graph)) * dS(2)
    #a += ufl.inner(ufl.grad(w), ufl.jump(u - utilde, n_graph)) * dS(2)
    #a += (sigma_v / ufl.avg(h_graph)) * ufl.inner(u - utilde, w - wtilde) * dS(2)
    

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
    L += ufl.inner(u_bc * n_graph, ufl.grad(w)) * ds
    L += (sigma_e / h_graph) * ufl.inner(u_bc, w) * ds

    problem = LinearProblem(
        a,
        L,
        bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="Poisson",
    )

    uh = problem.solve()

    if plotting:
        import pyvista as pv

        plotter = pv.Plotter()

        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["u"] = uh.x.array.real
        grid.set_active_scalars("u")
        plotter.add_mesh(grid, line_width=5, show_scalar_bar=True, cmap="viridis")

        plotter.view_xz()
        plotter.show_bounds()
        plotter.show()

    return uh, u_ex, dx

def L2(f: ufl.core.expr.Expr, dx: ufl.Measure) -> float:
    integral = ufl.inner(f, f) * dx
    return np.sqrt(assemble_scalar(integral, op=MPI.SUM))

if __name__ == "__main__":

    _, _, _ = Eval1DGraph(2, plotting=True)

    results = np.zeros((5, 3))

    for i in range(5):
        uh, u_ex, dx = Eval1DGraph(2**(i+1))
        results[i,0] = 2**(i+1)
        results[i,1] = L2(uh - u_ex, dx)
        if i > 0:
            results[i,2] = (np.log(results[i,1]) - np.log(results[i-1,1]))/(np.log(0.5))


    print(results)
