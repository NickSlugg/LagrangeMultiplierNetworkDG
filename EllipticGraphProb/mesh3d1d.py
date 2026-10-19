import numpy as np
import numpy.typing as npt
from typing import List
from mpi4py import MPI

import basix.ufl
import ufl
import dolfinx
import dolfinx.plot
import pyvista as pv

from mesh import GenMesh3D

class MeshData:
    # 3D Mesh Data
    nodes: npt.NDArray
    cells: npt.NDArray

    # Graph Mesh Data
    graph_nodes: npt.NDArray
    graph_cells: npt.NDArray
    marker_indices: npt.NDArray
    marker_values: npt.NDArray

    cell_indices: List[npt.NDArray]
    #edge_cells: npt.NDArray

    # Lagrange Multiplier Data
    lm_index: npt.NDArray
    lm_indices: npt.NDArray

class Mesh3D1D:
    _boundary_marker: int = 2
    _bifurc_marker: int = 1
    _comm: MPI.Intracomm

    # 3D Mesh Data
    _nodes: npt.NDArray
    _cells: npt.NDArray
    _mesh3D: dolfinx.mesh.Mesh

    # Graph Mesh Data
    _graph_nodes: npt.NDArray
    _graph_cells: npt.NDArray
    _graph_mesh: dolfinx.mesh.Mesh
    _global_facet_markers: dolfinx.mesh.MeshTags
    _marker_indices: npt.NDArray
    _marker_values: npt.NDArray
    _parent_vertex_marker: npt.NDArray

    _cell_indices: List[npt.NDArray]
    _edge_cells: npt.NDArray
    _edge_meshes: List[dolfinx.mesh.Mesh] | List = []
    _edge_entity_maps: List[dolfinx.mesh.EntityMap] | List = []
    _edge_vertex_maps: List[dolfinx.mesh.EntityMap] | List = []
    _edge_facet_markers: List[dolfinx.mesh.MeshTags] | List = []

    # Lagrange Multiplier Data
    _lm_index: npt.NDArray
    _lm_mesh: dolfinx.mesh.Mesh
    _lm_map: dolfinx.mesh.EntityMap
    _lm_indices: npt.NDArray
    _lm_to_edge_maps: List[dolfinx.mesh.EntityMap] | List = []

    def __init__(self, comm: MPI.Intracomm, mesh_data: MeshData):
        self._comm = comm

        self._nodes = mesh_data.nodes
        self._cells = mesh_data.cells

        self._graph_nodes = mesh_data.graph_nodes
        self._graph_cells = mesh_data.graph_cells
        self._marker_indices = mesh_data.marker_indices
        self._marker_values = mesh_data.marker_values

        self._cell_indices = mesh_data.cell_indices

        self._lm_index = mesh_data.lm_index
        self._lm_indices = mesh_data.lm_indices

        self.setup_graph_mesh()
        self.graph_mesh_edges()
        self.create_lm_maps()

    def setup_graph_mesh(self):

        # Create 3D Mesh
        ufl_cube = ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))

        self._mesh3D = dolfinx.mesh.create_mesh(
            self._comm,
            self._cells,
            ufl_cube,
            self._nodes,
        )

        # Generate Global Graph mesh
        ufl_int = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(3,)))

        self._graph_mesh = dolfinx.mesh.create_mesh(
            self._comm,
            self._graph_cells,
            ufl_int,
            self._graph_nodes,
        )

        # Create Lagrange Multiplier Submesh
        self._lm_mesh, self._lm_map = dolfinx.mesh.create_submesh(
            self._graph_mesh,
            self._graph_mesh.topology.dim - 1,
            self._lm_index
        )[:2]

        # Create facet marker for global graph
        self._global_facet_markers = dolfinx.mesh.meshtags(
            self._graph_mesh,
            0,
            self._marker_indices,
            self._marker_values,
        )

        parent_vertex_map = self._graph_mesh.topology.index_map(0)
        num_vertices_parent = parent_vertex_map.size_local + parent_vertex_map.num_ghosts
        self._parent_vertex_marker = np.full(num_vertices_parent, -1, dtype=np.int32)
        self._parent_vertex_marker[self._global_facet_markers.indices] = self._global_facet_markers.values

    def graph_mesh_edges(self):
        for i in range(len(self._cell_indices)):

            # create mesh for given edge
            edge_mesh, edge_map, vertex_map = dolfinx.mesh.create_submesh(
                self._graph_mesh,
                self._graph_mesh.topology.dim,
                self._cell_indices[i],
            )[:3]

            self._edge_meshes.append(edge_mesh)
            self._edge_entity_maps.append(edge_map)
            self._edge_vertex_maps.append(vertex_map)

            num_submesh_vertices = (
                edge_mesh.topology.index_map(0).size_local +
                edge_mesh.topology.index_map(0).num_ghosts
            )

            # Create facet marker on each edge
            parent_vertices = vertex_map.sub_topology_to_topology(
                np.arange(num_submesh_vertices, dtype=np.int32), inverse=False
            )

            sub_topology_values = self._parent_vertex_marker[parent_vertices]
            marked_vertices = np.flatnonzero(sub_topology_values >= 0)
            marked_values = sub_topology_values[marked_vertices].copy()

            self._edge_facet_markers.append(
                dolfinx.mesh.meshtags(edge_mesh, 0, marked_vertices.astype(np.int32), marked_values.astype(np.int32))
            )

    def create_lm_maps(self):
        for i, vertex_map in enumerate(self._edge_vertex_maps):
            lm_graph_indices = self._lm_map.sub_topology_to_topology(np.array([self._lm_indices[i]]), inverse=False)
            lm_to_edge = vertex_map.sub_topology_to_topology(lm_graph_indices, inverse=True)
            cpp_map = dolfinx.cpp.mesh.EntityMap(
                self._edge_meshes[i].topology._cpp_object,
                self._lm_mesh.topology._cpp_object,
                0,
                lm_to_edge,
            )
            self._lm_to_edge_maps.append(dolfinx.mesh.EntityMap(cpp_map))

    @property
    def boundary_marker(self):
        return self._boundary_marker

    @property
    def bifurc_marker(self):
        return self._bifurc_marker

    @property
    def comm(self):
        return self._comm

    @property
    def mesh3D(self):
        return self._mesh3D

    @property
    def graph_mesh(self):
        return self._graph_mesh

    @property
    def global_facet_markers(self):
        return self._global_facet_markers

    @property
    def edge_meshes(self):
        return self._edge_meshes

    @property
    def edge_entity_maps(self):
        return self._edge_entity_maps

    @property
    def edge_vertex_maps(self):
        return self._edge_vertex_maps

    @property
    def edge_facet_markers(self):
        return self._edge_facet_markers

    @property
    def lm_mesh(self):
        return self._lm_mesh

    @property
    def lm_map(self):
        return self._lm_map

    @property
    def lm_to_edge_maps(self):
        return self._lm_to_edge_maps

if __name__ == "__main__":

    N = 4 
    comm = MPI.COMM_WORLD

    nodes, cells, graph_nodes, graph_cells, cells_edges, lm_index, boundary_indices = GenMesh3D(N)

    marker_indices = np.insert(boundary_indices, 1, lm_index).astype(np.int32)

    mesh_data = MeshData()

    mesh_data.nodes = nodes
    #print(nodes)
    mesh_data.cells = cells

    mesh_data.graph_nodes = graph_nodes
    mesh_data.graph_cells = graph_cells
    mesh_data.marker_indices = marker_indices
    mesh_data.marker_values = np.array([2, 1, 2, 2], dtype=np.int32)

    num_cells_in_edge = N//2
    cells_array = np.arange(num_cells_in_edge, dtype=np.int32)
    cell_indices = []
    cell_indices.append(cells_array)
    cell_indices.append(2*cells_array + num_cells_in_edge)
    cell_indices.append(2*cells_array + 1 + num_cells_in_edge)

    mesh_data.cell_indices = cell_indices

    mesh_data.lm_index = lm_index
    mesh_data.lm_indices = np.array([num_cells_in_edge, 0, 0], dtype=np.int32)

    mesh_3d1d = Mesh3D1D(comm, mesh_data)

    mesh_3d1d.setup_graph_mesh()
    mesh_3d1d.graph_mesh_edges()
    mesh_3d1d.create_lm_maps()

    # Test if meshes created properly
    with dolfinx.io.VTKFile(mesh_3d1d.mesh3D.comm, "mesh_3d.pvd", "w") as vtk:
        vtk.write_mesh(mesh_3d1d.mesh3D)

    with dolfinx.io.VTKFile(mesh_3d1d.graph_mesh.comm, "graph_mesh.pvd", "w") as vtk:
        vtk.write_mesh(mesh_3d1d.graph_mesh)

    with dolfinx.io.VTKFile(mesh_3d1d.lm_mesh.comm, "lm_mesh.pvd", "w") as vtk:
        vtk.write_mesh(mesh_3d1d.lm_mesh)