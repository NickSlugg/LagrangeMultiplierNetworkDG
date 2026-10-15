import numpy as np
from mpi4py import MPI

import basix.ufl
import ufl
import dolfinx
import dolfinx.plot
import pyvista as pv

def GenMesh2D(N):
    m = N+1
    x_pts, y_pts = np.meshgrid(np.linspace(-1, 1, m), np.linspace(0, 2, m))
    nodes = np.stack([x_pts, y_pts], axis=-1).reshape(-1, 2)

    cells = np.zeros((2 * N **2, 3))
    n = N//2
    for i in range(n):
        for j in range(N):
            cells[2 * (j + N * i)] = np.array([0, 1, m]) + i + m * j
            cells[2 * (j + N * i) + 1] = np.array([1, m, m+1]) + i + m * j
            cells[2 * (j + N * i) + N ** 2] = np.array([n, m+n, m+n+1]) + i + m * j
            cells[2 * (j + N * i) + N ** 2 + 1] = np.array([n, n+1, m+n+1]) + i + m * j

    nodes = nodes.astype(np.float64)
    cells = cells.astype(np.int32)

    nodes1dindices = np.zeros((3 * (n + 1)  - 2,))
    nodes1dindices[:n+1] = m * np.arange(n+1) + n
    nodes1dindices[n+1:2*n+1] = (n+1) * N + N * np.arange(1, n+1)
    nodes1dindices[2*n+1:] = (n+1) * N + (N+2) * np.arange(1, n+1)

    graph_nodes = nodes[nodes1dindices.astype(np.int32)]

    graph_cells = np.repeat(np.arange(3*n+1), 2)[1:-1].reshape(3*n, 2)
    graph_cells[2*n, 0] = n

    return nodes, cells, graph_nodes, graph_cells.astype(np.int32)

def GenMesh3D(N, discontinuous=True):
    m = N+1
    n = N // 2
    x_pts, y_pts, z_pts = np.meshgrid(np.linspace(-1, 1, m), np.linspace(-1, 1, m), np.linspace(0, 2, m))
    nodes = np.stack([x_pts, y_pts, z_pts], axis=-1).reshape(-1, 3)
    nodes = np.flip(nodes, axis=1)

    cells = np.zeros((N**3, 8))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                cells[i + N * j + (N ** 2) * k] = np.array(
                    [0, 1, m**2, m**2 + 1, m, m + 1, m**2 + m, m**2 + m + 1]
                ) + i + m * j + (m ** 2) * k

    nodes = nodes.astype(np.float64)
    cells = cells.astype(np.int32)

    nodes1dindices = np.zeros((3 * (n + 1)  - 2,))

    nodes1dindices[:n+1] = m * np.arange(n+1) + n 
    nodes1dindices[n+1:2*n+1] = (n+1) * N + N * np.arange(1, n+1) 
    nodes1dindices[2*n+1:] = (n+1) * N + (N+2) * np.arange(1, n+1) 

    nodes1dindices += n * m ** 2

    graph_nodes = nodes[nodes1dindices.astype(np.int32)]

    graph_cells = np.repeat(np.arange(3*n+1), 2)[1:-1].reshape(3*n, 2)
    graph_cells[2*n, 0] = n

    if discontinuous:
        edges = []
        #    nodes[nodes1dindices[:n+1].astype(np.int32)],
        #    nodes[nodes1dindices[n:2*n+1].astype(np.int32)],
        #    nodes[nodes1dindices[np.concatenate([nodes1dindices[n]], nodes1dindices[2*n+1:]).astype(np.int32)]]
        #]

        cells_edges = [
            graph_cells[:n],
            graph_cells[n:2*n],
            graph_cells[2*n:],
        ]

        lm_index = np.array([n]) + n * (m ** 2)
        boundary_indices = np.array([0, 3*n-1, 3*n]) + n * (m ** 2)

        return nodes, cells, graph_nodes, graph_cells, cells_edges, lm_index, boundary_indices

    return nodes, cells, graph_nodes, graph_cells

if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf)

    N = 2
    comm = MPI.COMM_WORLD
    nodes, cells, graph_nodes, graph_cells, cells_edges, lm_index, boundary_indices = GenMesh3D(2*N)

    #print(nodes)
    #print(cells)

    ufl_cube = ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))
    ufl_int = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(3,)))

    mesh_3d = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, cells=cells, x=nodes, e=ufl_cube
    )

    mesh_1d = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, cells=graph_cells, x=graph_nodes, e=ufl_int
    )

    topology3D, cell_types3D, geometry3D = dolfinx.plot.vtk_mesh(mesh_3d)
    topology1D, cell_types1D, geometry1D = dolfinx.plot.vtk_mesh(mesh_1d)

    grid = pv.UnstructuredGrid(topology3D, cell_types3D, geometry3D)
    grid1D = pv.UnstructuredGrid(topology1D, cell_types1D, geometry1D)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_mesh(grid1D, show_edges=True, color="red", line_width=2)
    plotter.show_bounds()
    plotter.show()

    """
    graph_mesh = dolfinx.mesh.create_mesh(
        comm,
        e=ufl_int,
        cells=graph_nodes,
        x=graph_cells,
    )

    edge_meshs = {}

    for i, edge in enumerate(edges):
        edge_meshs[i] = dolfinx.mesh.create_submesh(
            graph_mesh,
            graph_mesh.topology.dim,
            cells_edges[i]
        )[:2]

    lm_mesh, lm_map = dolfinx.mesh.create_submesh(
        graph_mesh,
        graph_mesh.topology.dim - 1,
        np.array([N // 2]),
    )[:2]

    topology1D, cell_types1D, geometry1D = dolfinx.plot.vtk_mesh(mesh_1d)

    plotter = pv.Plotter()
    plotter.add_mesh(grid1D, show_edges=True, color="red", line_width=2)
    plotter.show_bounds()
    plotter.show()


    nodes, cells, graph_nodes, graph_cells = GenMesh(4)

    ufl_tri = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    ufl_int = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(2,)))

    mesh_3d = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, cells=cells, x=nodes, e=ufl_tri
    )

    mesh_1d = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD, cells=graph_cells, x=graph_nodes, e=ufl_int
    )

    topology3D, cell_types3D, geometry3D = dolfinx.plot.vtk_mesh(mesh_3d)
    topology1D, cell_types1D, geometry1D = dolfinx.plot.vtk_mesh(mesh_1d)

    grid = pv.UnstructuredGrid(topology3D, cell_types3D, geometry3D)
    grid1D = pv.UnstructuredGrid(topology1D, cell_types1D, geometry1D)
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_mesh(grid1D, show_edges=True, color="red", line_width=2)
    plotter.show_bounds()
    plotter.show()
    """

