# import packages
import numpy as np
import numpy.typing as npt
from typing import List

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.plot
import ufl
import basix.ufl

from fenicsx_ii import assemble_scalar

from mesh import GenMesh3D

def L22(f: ufl.core.expr.Expr, dx: ufl.Measure):
  integral = ufl.inner(f, f) * dx
  return assemble_scalar(integral, op=MPI.SUM)

class GraphProblem:
    _boundary_marker:int = 2
    _bifurc_marker:int = 1
    _comm: MPI.Intracomm = MPI.COMM_WORLD

    # Mesh Data
    _graph_nodes: npt.NDArray
    _graph_cells: npt.NDArray
    _cells_edges: List
    _lm_index: npt.NDArray
    _boundary_indices: npt.NDArray
    _graph_mesh: dolfinx.mesh.Mesh
    _lm_mesh: dolfinx.mesh.Mesh
    _lm_map: dolfinx.mesh.EntityMap
    _marker_indices: npt.NDArray
    _marker_values: npt.NDArray = np.array([2, 1, 2, 2], dtype=np.int32)
    _parent_vertex_marker: npt.NDArray

    # FS Data
    _edge_spaces: List[dolfinx.fem.FunctionSpace]
    _lm_space: dolfinx.fem.FunctionSpace
    _function_spaces: ufl.MixedFunctionSpace

    # Functions
    _u_e: List[ufl.Argument]
    _u_tilde: ufl.Argument
    _w_e: List[ufl.Argument]
    _w_tilde: ufl.Argument
    _functions: List | None = None

    # data

    # DG Data
    _sigma_edge = 30
    _sigma_bifurc = 30


    def __init__(self):
        self._edge_meshes = []
        self._edge_entity_maps = []
        self._edge_vertex_maps = []
        self._edge_facet_markers = []

    def global_graph_mesh(self, N: int):

        # Set Graph Data
        _, _, self._graph_nodes, self._graph_cells, self._cells_edges, self._lm_index, self._boundary_indices = GenMesh3D(2*N)

        # Generate global graph mesh
        ufl_int = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(3,)))

        self._graph_mesh = dolfinx.mesh.create_mesh(
            self._comm,
            e=ufl_int,
            cells=self._graph_cells,
            x=self._graph_nodes,
        )

        # submesh for lagrange multiplier(s)
        self._lm_mesh, self._lm_map = dolfinx.mesh.create_submesh(
            self._graph_mesh,
            self._graph_mesh.topology.dim - 1,
            self._lm_index
        )[:2]

        # set marker for boundary/bifurcation nodes
        self._marker_indices = np.insert(self._boundary_indices, 1, self._lm_index).astype(np.int32)
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

        num_cells_in_edge = len(self._cells_edges[0])
        cells_array = np.arange(num_cells_in_edge, dtype=np.int32)
        cell_indices = []
        cell_indices.append(cells_array)
        cell_indices.append(2*cells_array + num_cells_in_edge)
        cell_indices.append(2*cells_array + 1 + num_cells_in_edge)


        for i in range(len(self._cells_edges)):

            edge_mesh, edge_map, vertex_map = dolfinx.mesh.create_submesh(
                self._graph_mesh,
                self._graph_mesh.topology.dim,
                cell_indices[i],
            )[:3]

            self._edge_meshes.append(edge_mesh)
            self._edge_entity_maps.append(edge_map)
            self._edge_vertex_maps.append(vertex_map)

            num_submesh_vertices = (
                edge_mesh.topology.index_map(0).size_local +
                edge_mesh.topology.index_map(0).num_ghosts
            )
            parent_vertices = vertex_map.sub_topology_to_topology(
                np.arange(num_submesh_vertices, dtype=np.int32), inverse=False
            )
            sub_topology_values = self._parent_vertex_marker[parent_vertices]
            marked_vertices = np.flatnonzero(sub_topology_values >= 0)
            marked_values = sub_topology_values[marked_vertices].copy()

            self._edge_facet_markers.append(
                dolfinx.mesh.meshtags(edge_mesh, 0, marked_vertices.astype(np.int32), marked_values.astype(np.int32))
            )

    def create_function_spaces(self, degree=1):

        edge_element = basix.ufl.element(
            family="Discontinuous Lagrange",
            cell="interval",
            degree=degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
        )

        self._edge_spaces = [dolfinx.fem.functionspace(edge_mesh, edge_element) for edge_mesh in self._edge_meshes]
        self._lm_space = dolfinx.fem.functionspace(self._lm_mesh, ("DG", 0))

        self._function_spaces = ufl.MixedFunctionSpace(*[*self._edge_spaces, self._lm_space])

        self._trial_fun = ufl.TrialFunctions(self._function_spaces)
        self._u_e = self._trial_fun[:-1]
        self._u_tilde = self._trial_fun[-1]

        self._test_fun = ufl.TestFunctions(self._function_spaces)
        self._w_e = self._test_fun[:-1]
        self._w_tilde = self._test_fun[-1]

        # set entity maps to edge functions
        self._lm_to_edge_maps = []
        lm_indices = np.array([0], dtype=np.int32)

        for i, vertex_map in enumerate(self._edge_vertex_maps):
            lm_graph_indices = self._lm_map.sub_topology_to_topology(lm_indices, inverse=False)
            lm_to_edge = vertex_map.sub_topology_to_topology(lm_graph_indices, inverse=True)
            cpp_map = dolfinx.cpp.mesh.EntityMap(
                self._edge_meshes[i].topology._cpp_object,
                self._lm_mesh.topology._cpp_object,
                0,
                lm_to_edge,
            )
            self._lm_to_edge_maps.append(dolfinx.mesh.EntityMap(cpp_map))

    def set_exact_soln(self, u_ex_edges, f_edges):
        self._f_edges = f_edges
        self._u_ex_edges = u_ex_edges

    def setup_forms(self):

        a = [
            [ufl.ZeroBaseForm((ui, vj)) for vj in self._test_fun] for ui in self._trial_fun
        ]
        L = [
            ufl.ZeroBaseForm((ui,)) for ui in self._test_fun
        ]


        dS_global = ufl.Measure("dS", domain=self._graph_mesh, subdomain_data=self._global_facet_markers)

        # Redefine forms to ensure they are properly categorized
        for i, (edge, facet_marker) in enumerate(zip(self._edge_meshes, self._edge_facet_markers)):
            dx_edge = ufl.Measure("dx", domain=edge)
            dS_edge = ufl.Measure("dS", domain=edge)
            ds_edge = ufl.Measure("ds", domain=edge, subdomain_data=facet_marker)

            dx_bifurc = ufl.Measure("dx", domain=self._lm_mesh)

            # Use a constant scalar for h_e inside the 0D term to avoid compiling 1D variables on a 0D domain
            # We calculate the physical cell size (assuming uniform intervals in the submesh)
            coords = edge.geometry.x
            h_e_val = float(np.linalg.norm(coords[1] - coords[0]))

            h_e = ufl.CellDiameter(edge)
            h_e_avg = (h_e("+") + h_e("-"))/2
            n_e = ufl.FacetNormal(edge)

            # DG Form - Diagonal blocks
            a[i][i] += ufl.inner(ufl.grad(self._u_e[i]), ufl.grad(self._w_e[i])) * dx_edge
            a[i][i] -= ufl.inner(ufl.avg(ufl.grad(self._u_e[i])), ufl.jump(self._w_e[i], n_e)) * dS_edge
            a[i][i] -= ufl.inner(ufl.avg(ufl.grad(self._w_e[i])), ufl.jump(self._u_e[i], n_e)) * dS_edge
            a[i][i] += (self._sigma_edge / h_e_avg) * ufl.inner(ufl.jump(self._u_e[i], n_e), ufl.jump(self._w_e[i], n_e)) * dS_edge
            a[i][i] -= ufl.inner(ufl.grad(self._u_e[i]), ufl.outer(self._w_e[i], n_e)) * ds_edge(self._boundary_marker)
            a[i][i] -= ufl.inner(ufl.grad(self._w_e[i]), ufl.outer(self._u_e[i], n_e)) * ds_edge(self._boundary_marker)
            a[i][i] += (self._sigma_edge / h_e) * ufl.inner(self._u_e[i], self._w_e[i]) * ds_edge(self._boundary_marker)

            # RHS Form
            L[i] += ufl.inner(self._w_e[i], self._f_edges[i]) * dx_edge
            L[i] -= ufl.inner(ufl.grad(self._w_e[i]), ufl.outer(self._u_ex_edges[i], n_e)) * ds_edge(self._boundary_marker)
            L[i] += (self._sigma_edge / h_e) * ufl.inner(self._u_ex_edges[i], self._w_e[i]) * ds_edge(self._boundary_marker)

            # Bifurcation node coupling
            a[i][i] += ufl.inner(ufl.grad(self._u_e[i]), ufl.outer(self._w_e[i], n_e)) * ds_edge(self._bifurc_marker)
            a[i][i] += ufl.inner(ufl.grad(self._w_e[i]), ufl.outer(self._u_e[i], n_e)) * ds_edge(self._bifurc_marker)
            a[i][i] += (self._sigma_bifurc / h_e) * ufl.inner(self._u_e[i], self._w_e[i]) * ds_edge(self._bifurc_marker)

            a[-1][i] -= ufl.inner(ufl.grad(self._u_e[i]), ufl.outer(self._w_tilde, n_e)) * ds_edge(self._bifurc_marker)
            a[i][-1] -= ufl.inner(ufl.grad(self._w_e[i]), ufl.outer(self._u_tilde, n_e)) * ds_edge(self._bifurc_marker)

            a[-1][i] -= (self._sigma_bifurc / h_e) * ufl.inner(self._u_e[i], self._w_tilde) * ds_edge(self._bifurc_marker)
            a[i][-1] -= (self._sigma_bifurc / h_e) * ufl.inner(self._w_e[i], self._u_tilde) * ds_edge(self._bifurc_marker)

            # Use h_e_val (the scalar float) here instead of the UFL object h_e
            a[-1][-1] += (self._sigma_bifurc / h_e_val) * self._u_tilde * self._w_tilde * dx_bifurc

        entity_maps = self._edge_entity_maps + [self._lm_map] + self._lm_to_edge_maps

        for i, ai in enumerate(a):
            for j, aij in enumerate(ai):
                if isinstance(aij, ufl.ZeroBaseForm):
                    a[i][j] = None

        self._L_form = dolfinx.fem.form(L, entity_maps=entity_maps)
        self._a_form = dolfinx.fem.form(a, entity_maps=entity_maps)

    def solve_system(self):
        
        # Create and assemble the block matrix using standard dolfinx API
        A_mat = dolfinx.fem.petsc.create_matrix(self._a_form)
        A_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A_mat, self._a_form, bcs=[])
        A_mat.assemble()

        # Extract function spaces and create the block RHS vector
        spaces = dolfinx.fem.extract_function_spaces(self._L_form)
        kind = "nest" if A_mat.getType() == "nest" else None

        b_vec = dolfinx.fem.petsc.create_vector(spaces, kind=kind)
        with b_vec.localForm() as loc_b:
            loc_b.set(0)
        dolfinx.fem.petsc.assemble_vector(b_vec, self._L_form)
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Create block solution vector
        x = dolfinx.fem.petsc.create_vector(spaces, kind=kind)

        # Configure and run the PETSc solver directly
        ksp = PETSc.KSP().create(comm=self._comm)
        ksp.setOperators(A_mat)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

        ksp.solve(b_vec, x)

        self._functions = []

        for i, edge in enumerate(self._edge_spaces):
            self._functions.append(dolfinx.fem.Function(edge, name=f"edge_{i}"))
        self._functions.append(dolfinx.fem.Function(self._lm_space, name="lm_value"))

        dolfinx.la.petsc._ghost_update(
            x,
            insert_mode=PETSc.InsertMode.INSERT,
            scatter_mode=PETSc.ScatterMode.FORWARD,
        )

        dolfinx.fem.petsc.assign(x, self._functions)

        return self._functions

    def compute_error(self, u_ex):
        assert self._functions is not None

        squared_norm = 0
        lm_diffs = np.zeros((len(self._edge_meshes),))
        f_tilde = self._functions[-1]
        dx_bifurc = ufl.Measure("dx", domain=self._lm_mesh)
        for i, (f, edge) in enumerate(zip(self._functions[:-1], self._edge_meshes)):
            dx_edge = ufl.Measure("dx", domain=edge)
            dS_edge = ufl.Measure("dS", domain=edge)
            ds_edge = ufl.Measure("ds", domain=edge, subdomain_data=self._edge_facet_markers[i])

            h_e = ufl.CellDiameter(edge)
            h_e_avg = (h_e("+") + h_e("-"))/2

            squared_int = ufl.inner(f - u_ex[i], f - u_ex[i]) * dx_edge 
                #+ (self._sigma_edge / h_e_avg) * ufl.inner(ufl.jump(f - u_ex[i]), ufl.jump(f - u_ex[i])) * dS_edge \
                #+ (self._sigma_edge / h_e) * ufl.inner(f - u_ex[i], f - u_ex[i]) * ds_edge(self._boundary_marker) \
                #+ (self._sigma_bifurc / h_e) * ufl.inner(f - u_ex[i], f - u_ex[i]) * ds_edge(self._bifurc_marker) \
                #- 2 * (self._sigma_bifurc / h_e) * ufl.inner(f - u_ex[i], f_tilde - u_ex[-1]) * ds_edge(self._bifurc_marker) \
                #+ (self._sigma_bifurc / h_e) * ufl.inner(f_tilde - u_ex[-1], f_tilde - u_ex[-1]) * dx_bifurc

            entity_maps = self._edge_entity_maps + [self._lm_map] + self._lm_to_edge_maps

            int_form = dolfinx.fem.form(squared_int)
            #scalar_int = dolfinx.fem.assemble_scalar(dolfinx.fem.form(squared_int))
            squared_norm += assemble_scalar(squared_int, op=MPI.SUM)
            #lm_diffs[i] = assemble_scalar(lm_form, op=MPI.SUM)

        return np.sqrt(squared_norm)

    def reset_edges(self):
        self._edge_meshes = []
        self._edge_entity_maps = []
        self._edge_vertex_maps = []
        self._edge_facet_markers = []

    @property
    def boundary_marker(self):
        return self._boundary_marker

    @property
    def bifurc_marker(self):
        return self._bifurc_marker

    @property
    def graph_mesh(self):
        return self._graph_mesh

    @property
    def lm_mesh(self):
        return self._lm_mesh

    @property
    def edge_meshes(self):
        return self._edge_meshes

    @property
    def edge_spaces(self):
        return self._edge_spaces

    @property
    def functions(self):
        return self._functions
