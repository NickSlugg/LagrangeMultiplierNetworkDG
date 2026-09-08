import logging
import typing

from petsc4py import PETSc

import numpy as np
import numpy.typing as npt

## FEniCSx Packages
import basix
import ufl
from ufl import dot, grad, inner, jump, avg, CellDiameter
import dolfinx.la.petsc as _petsc_la
from dolfinx import common, fem

# Networkx/Fenics-networks packages
from networks_fenicsx import NetworkMesh

def compute_integration_data(network_mesh:NetworkMesh) -> tuple[dict[int, npt.NDArray[np.int32]], dict[int, npt.NDArray[np.int32]]]:

    # write network bifurcation values to dictionary
    influx_color_to_bifurcations: dict[int, npt.NDArray[np.int32]] = {
        int(color): np.empty(0, dtype=np.int32) for color in range(network_mesh.num_edge_colors)
    }
    outflux_color_to_bifurcations: dict[int, npt.NDArray[np.int32]] = {
        int(color): np.empty(0, dtype=np.int32) for color in range(network_mesh.num_edge_colors)
    }

    for i, bifurcation in enumerate(network_mesh.bifurcation_values):
        for color in network_mesh.in_edges(i):
            influx_color_to_bifurcations[color] = np.append(
                influx_color_to_bifurcations[color], bifurcation
            )
            outflux_color_to_bifurcations[color] = np.append(
                outflux_color_to_bifurcations[color], bifurcation
            )

    # initialize entities dictionaries
    in_flux_entities: dict[int, npt.NDArray[np.int32]] = {}
    out_flux_entities: dict[int, npt.NDArray[np.int32]] = {}

    #
    for color in range(network_mesh.num_edge_colors):
        sm = network_mesh.submeshes[color]
        smfm = network_mesh.submesh_facet_markers[color]
        sm.topology.create_connectivity(sm.topology.dim - 1, sm.topology.dim)

        # compute influx entities
        submesh_influx_entities = fem.compute_integration_domains(
            fem.IntegralType.exterior_facet,
            sm.topology,
            smfm.indices[np.isin(smfm.values, influx_color_to_bifurcations[color])],
        ).reshape(-1,2)
        parent_to_sub_influx = network_mesh.entity_maps[color].sub_topology_to_topology(
            submesh_influx_entities[:,0].copy(), inverse=False
        )
        submesh_influx_entities[:,0] = parent_to_sub_influx
        in_flux_entities[color] = submesh_influx_entities.flatten()

        # compute outflux entities
        submesh_outflux_entities = fem.compute_integration_domains(
            fem.IntegralType.exterior_facet,
            sm.topology,
            smfm.indices[np.isin(smfm.values, outflux_color_to_bifurcations[color])],
        ).reshape(-1,2)
        parent_to_sub_outflux = network_mesh.entity_maps[color].sub_topology_to_topology(
            submesh_outflux_entities[:,0].copy(), inverse=False
        )
        submesh_outflux_entities[:,0] = parent_to_sub_outflux
        out_flux_entities[color] = submesh_outflux_entities.flatten()

    return in_flux_entities, out_flux_entities



class PoissonAssembler:
    """
    Assembles forms for the variational problem

    -d_s (k d_s u_e) = f_e

    on e \in G. 
    """

    _network_mesh: NetworkMesh # network mesh
    _edge_spaces: list[fem.FunctionSpace] # function spaces for each edge
    _lm_space: fem.FunctionSpace # function space for lagrange multipliers
    _a: list[list[fem.Form]] # array for bilinear forms
    _L: list[fem.Form] # array for linear forms

    def __init__(self, mesh: NetworkMesh, degree=1):
        self._network_mesh = mesh
        submeshes = self._network_mesh.submeshes

        # define function space on each edge
        edge_element = basix.ufl.element(
            family="Lagrange",
            cell="interval",
            degree=degree,
            lagrange_variant=basix.LagrangeVariant.equispaced,
            discontinuous=True
        )
        self._edge_spaces = [fem.functionspace(submsh, edge_element) for submsh in submeshes]   # 1d Solutions

        self._lm_space = fem.functionspace(self._network_mesh.lm_mesh, ("DG", 0))   # Lagrange Multipliers

        # compute integration data
        self._integration_data = []
        self._in_idx = max(mesh.in_marker, mesh.out_marker) + 1
        in_flux_entities, out_flux_entities = compute_integration_data(self._network_mesh)
        self._in_keys = tuple(in_flux_entities.keys())
        self._out_keys = tuple(out_flux_entities.keys())

        for color in self._in_keys:
            self._integration_data.append((self._in_idx + color, in_flux_entities[color]))
        self._out_idx = self._in_idx + len(out_flux_entities)
        for color in self._out_keys:
            self._integration_data.append((self._out_idx + color, out_flux_entities[color]))

    def compute_forms(self, 
        u_bc_ex: fem.Expression | ufl.core.expr.Expr | None = None, 
        f_hat: ufl.core.expr.Expr | None = None,
        kappa: ufl.core.expr.Expr | None = None,
        jit_options: dict | None = None,
        form_compiler_options: dict | None = None
        ):
        """
        Computer the forms for the 3D - 1D problem with lagrange multipliers.

        Args:
        f_hat: source term
        u_bc_ex: boundary conditions
        kappa: diffusion parameter
        """
        num_edges = self._network_mesh.num_edge_colors

        # Trial and test functions for each component
        test_functions = [ufl.TestFunction(fs) for fs in self.function_spaces]
        trial_functions = [ufl.TrialFunction(fs) for fs in self.function_spaces]

        # initialize bilinear/linear forms
        a: list[list[ufl.Form | ufl.ZeroBaseForm | None]] = [
            [ufl.ZeroBaseForm((ui, vj)) for vj in test_functions] for ui in trial_functions
        ]
        L: list[ufl.Form | ufl.ZeroBaseForm | None] = [
            ufl.ZeroBaseForm((ui,)) for ui in test_functions
        ]

        # Default State Values
        if f_hat is None:
            f_hat = fem.Constant(self._network_mesh.mesh, 0.0)
        if u_bc_ex is None:
            u_bc_ex = fem.Constant(self._network_mesh.mesh, 0.0)
        if kappa is None:
            kappa = fem.Constant(self._network_mesh.mesh, 1.0)

        # Store trial/test_functions for edges:
        u_trial, w_test = [], []
        for edge in self._edge_spaces:
            u_trial.append(ufl.TrialFunction(edge))
            w_test.append(ufl.TestFunction(edge))

        # Lagrange Multipliers
        u_tilde = ufl.TrialFunction(self.lm_space)
        w_tilde = ufl.TestFunction(self.lm_space)

        # assemble boundary condition function
        fs_global = fem.functionspace(self._network_mesh.mesh, ("DG", 1))
        u_bc = fem.Function(fs_global)
        if u_bc_ex is None:
            u_bc.interpolate(fem.Constant(self._network_mesh.mesh, 0.0))
            """elif isinstance(u_bc_ex, ufl.core.expr.Expr):
                try:
                    expr = fem.Expression(u_bc_ex, fs_global.element.interpolation_points())
                except TypeError:
                    expr = fem.Expression(u_bc_ex, fs_global.element.interpolation_points)
                u_bc.interpolate(u_bc_ex)
            else:
                u_bc.interpolate(u_bc_ex)
            """
        else:
            u_bc.interpolate(lambda x: np.full((u_bc.function_space.dofmap.index_map.size_local,), 0.0))

        sigma_edges = 30
        sigma_bifurc = 30

        dx_global = ufl.Measure("dx", domain=self._network_mesh.mesh)

        J = ufl.Jacobian(self._network_mesh.mesh)
        t = J[:, 0]
        t /= ufl.sqrt(ufl.inner(t, t))

        tangent = self._network_mesh.orientation * t

        ## Assemble form components
        for i, (submesh, entity_map, facet_marker) in enumerate(
            zip(self._network_mesh.submeshes, self._network_mesh.entity_maps, self._network_mesh.submesh_facet_markers)
        ):
            dx_edge = ufl.Measure("dx", domain=submesh)
            dS_edge = ufl.Measure("dS", domain=submesh)
            ds_edge = ufl.Measure("ds", domain=submesh, subdomain_data = facet_marker)

            h_e = CellDiameter(submesh)

            a[i][i] += kappa * dot(grad(u_trial[i]), tangent) * dot(grad(w_test[i]), tangent) * dx_edge \
            - avg(kappa * dot(grad(u_trial[i]), tangent)) * jump(grad(w_test[i]), tangent) * dS_edge \
            - avg(kappa * dot(grad(w_test[i]), tangent)) * jump(grad(u_trial[i]), tangent) * dS_edge \
            + sigma_edges / avg(h_e) * inner( jump(u_trial[i], tangent), jump(w_test[i], tangent)) * dS_edge \
            - kappa * dot(grad(u_trial[i]), tangent) * w_test[i] * ds_edge(self._network_mesh.in_marker)
            - kappa * dot(grad(w_test[i]), tangent) * u_trial[i] * ds_edge(self._network_mesh.in_marker)
            + sigma_edges / h_e * u_trial[i] * w_test[i] * ds_edge(self._network_mesh.in_marker)
            + kappa * dot(grad(u_trial[i]), tangent) * w_test[i] * ds_edge(self._network_mesh.out_marker)
            + kappa * dot(grad(w_test[i]), tangent) * u_trial[i] * ds_edge(self._network_mesh.out_marker)
            + sigma_edges / h_e * u_trial[i] * w_test[i] * ds_edge(self._network_mesh.in_marker)
            - kappa * dot(grad(u_trial[i]), tangent) * w_test[i] * ds_edge(self._network_mesh.in_marker) \
            + kappa * dot(grad(u_trial[i]), tangent) * w_test[i] * ds_edge(self._network_mesh.out_marker) \
            - kappa * dot(grad(w_test[i]), tangent) * u_trial[i] * ds_edge(self._network_mesh.in_marker) \
            + kappa * dot(grad(w_test[i]), tangent) * u_trial[i] * ds_edge(self._network_mesh.out_marker) \
            + sigma_bifurc / h_e * u_trial[i] * w_test[i] * ds_edge

            L[i] += w_test[i] * f_hat * dx_edge
            L[i] -= kappa * dot(grad(w_test[i]), tangent) * u_bc * ds_edge(self._network_mesh.in_marker)
            L[i] += sigma_edges / h_e * u_bc * w_test[i] * ds_edge(self._network_mesh.in_marker)
            L[i] += kappa * dot(grad(w_test[i]), tangent) * u_bc * ds_edge(self._network_mesh.out_marker)
            L[i] += sigma_edges / h_e * u_bc * w_test[i] * ds_edge(self._network_mesh.out_marker)

        ## Assemble Lagrange multiplier components
        ds = ufl.Measure("ds", domain=self._network_mesh.mesh, subdomain_data=self._integration_data)
        for color in self._in_keys:
            a[-1][color] += kappa * dot(grad(u_trial[color]), tangent) * w_tilde * ds(self._in_idx + color)
            a[-1][color] -= sigma_bifurc / h_e * u_trial[color] * w_tilde * ds(self._in_idx + color)
            a[color][-1] += kappa * dot(grad(w_test[color]), tangent) * u_tilde * ds(self._in_idx + color)
            a[color][-1] -= sigma_bifurc / h_e * w_test[color] * u_tilde * ds(self._in_idx + color)
            a[-1][-1] += sigma_bifurc / h_e * u_tilde * w_tilde * ds(self._in_idx + color)

        for color in self._in_keys:
            a[-1][color] -= kappa * dot(grad(u_trial[color]), tangent) * w_tilde * ds(self._out_idx + color)
            a[-1][color] -= sigma_bifurc / h_e * u_trial[color] * w_tilde * ds(self._out_idx + color)
            a[color][-1] -= kappa * dot(grad(w_test[color]), tangent) * u_tilde * ds(self._out_idx + color)
            a[color][-1] -= sigma_bifurc / h_e * w_test[color] * u_tilde * ds(self._out_idx + color)
            a[-1][-1] += sigma_bifurc / h_e * u_tilde * w_tilde * ds(self._out_idx + color)

        # set up entity maps
        entity_maps = [entity_map, self._network_mesh.lm_map, *self._network_mesh.entity_maps]

        # replace remaining zerobaseforms with none
        for i, ai in enumerate(a):
            for j, aij in enumerate(ai):
                if isinstance(aij, ufl.ZeroBaseForm):
                    a[i][j] = None

        # setup up forms
        self._a = fem.form(
            a,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )
        self._L = fem.form(
            L,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )

    def assemble(self, A: PETSc.Mat | None = None, b: PETSc.Mat | None = None, assemble_lhs: bool = True, assemble_rhs: bool = True, kind: str | typing.Sequence[typing.Sequence[str]] | None = None) -> tuple[PETSc.Mat, PETSc.Vec]:
        if assemble_lhs:
            if A is None:
                A = fem.petsc.create_matrix([[aij for aij in ai] for ai in self._a], kind=kind)
            A = fem.petsc.assemble_matrix(A, self._a, bcs=[])
            A.assemble()
            kind = "nest" if A.getType() == PETSc.Mat.Type.NEST else kind
        if assemble_rhs:
            if b is None:
                assert isinstance(kind, str) or kind is None
                b = fem.petsc.create_vector(fem.extract_function_spaces(self._L), kind=kind)
            b = fem.petsc.assemble_vector(b, self._L)
            _petsc_la._ghost_update(
                b, 
                insert_mode=PETSc.InsertMode.ADD_VALUES,
                scatter_mode=PETSc.ScatterMode.REVERSE
            )
        return (A, b)

    @property
    def function_spaces(self) -> list[fem.FunctionSpace]:
        """Function spaces for bilinear form"""
        return [*self._edge_spaces, self._lm_space]

    @property
    def network(self) -> NetworkMesh:
        return self._network_mesh

    @property
    def lm_space(self) -> fem.FunctionSpace:
        return self._lm_space

    @property
    def edge_spaces(self) -> list[fem.FunctionSpace]:
        return self._edge_spaces

    @property
    def bilinear_forms(self) -> typing.Sequence[typing.Sequence[fem.Form]]:
        if self._a is None:
            logging.error("Bilinear forms have not been computed. Must call compute_forms()")
        else:
            return self._a

    @property
    def linear_forms(self) -> typing.Sequence[typing.Sequence[fem.Form]]:
        if self._L is None:
            logging.error("Bilinear forms have not been computed. Must call compute_forms()")
        else:
            return self._L


        