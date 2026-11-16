import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.plot
import ufl
import basix.ufl

from mesh import GenMesh3D, GenMesh3DPerp
from mesh3d1d import MeshData, Mesh3D1D

from fenicsx_ii import Average, Circle, create_matrix, assemble_matrix, create_vector, assemble_vector, assemble_scalar

# function for solving system

def solve_3d1d(N, comm=MPI.COMM_WORLD):

    # generate 3d1d mesh
    nodes, cells, graph_nodes, graph_cells, cells_edges, lm_index, boundary_indices = GenMesh3DPerp(N)

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

    # create functionspaces
    fs_3d = dolfinx.fem.functionspace(mesh_3d1d.mesh3D, ("DG", 1))
    fs_edges = [dolfinx.fem.functionspace(edge_mesh, ("DG", 1)) for edge_mesh in mesh_3d1d.edge_meshes]
    fs_lm = dolfinx.fem.functionspace(mesh_3d1d.lm_mesh, ("DG", 0))

    function_spaces = ufl.MixedFunctionSpace(*[fs_3d, *fs_edges, fs_lm])

    trial_funs = ufl.TrialFunctions(function_spaces)
    u_3d = trial_funs[0]
    u_1d = trial_funs[1:-1]
    u_lm = trial_funs[-1]

    test_funs = ufl.TestFunctions(function_spaces)
    w_3d = test_funs[0]
    w_1d = test_funs[1:-1]
    w_lm = test_funs[-1]    

    # rhs functions
    f_omega = dolfinx.fem.Constant(mesh_3d1d.mesh3D, 1.0)

    f_edges = [
        1,
        1,
        1,
    ]

    # assemble bilinear and linear forms
    xi = 1
    radius = 0.05
    area = ufl.pi * radius ** 2
    perimeter = 2 * ufl.pi * radius

    q_degree = 20

    sigma_omega = 20
    sigma_e = 20
    sigma_bifurc = 20

    # initialize forms
    a = [
        [ufl.ZeroBaseForm((ui, vj)) for vj in test_funs] for ui in trial_funs
    ]
    L = [
        ufl.ZeroBaseForm((ui,)) for ui in test_funs
    ]

    # 3D component

    # measures
    dx = ufl.Measure("dx", domain=mesh_3d1d.mesh3D)
    dS = ufl.Measure("dS", domain=mesh_3d1d.mesh3D)
    ds = ufl.Measure("ds", domain=mesh_3d1d.mesh3D)

    h_omega = ufl.CellDiameter(mesh_3d1d.mesh3D)
    h_omega_avg = (h_omega("+") + h_omega("-"))/2
    n_omega = ufl.FacetNormal(mesh_3d1d.mesh3D)

    # Bilinear Form
    a[0][0] += ufl.inner(ufl.grad(u_3d), ufl.grad(w_3d)) * dx
    a[0][0] -= ufl.inner(ufl.avg(ufl.grad(u_3d)), ufl.jump(w_3d, n_omega)) * dS
    a[0][0] -= ufl.inner(ufl.avg(ufl.grad(w_3d)), ufl.jump(u_3d, n_omega)) * dS
    a[0][0] += (sigma_omega / h_omega_avg) * ufl.inner(ufl.jump(u_3d, n_omega), ufl.jump(w_3d, n_omega)) * dS
    a[0][0] -= ufl.inner(ufl.grad(u_3d), w_3d * n_omega) * ds
    a[0][0] -= ufl.inner(ufl.grad(w_3d), u_3d * n_omega) * ds
    a[0][0] += (sigma_omega / h_omega) * ufl.inner(u_3d, w_3d) * ds

    # Linear Form
    L[0] += f_omega * w_3d * dx

    bifurc_marker = mesh_3d1d.bifurc_marker

    # Edge components
        # Redefine forms to ensure they are properly categorized
    for i, (edge, facet_marker) in enumerate(zip(mesh_3d1d.edge_meshes, mesh_3d1d.edge_facet_markers)):
        dx_edge = ufl.Measure("dx", domain=edge)
        dS_edge = ufl.Measure("dS", domain=edge)
        ds_edge = ufl.Measure("ds", domain=edge, subdomain_data=facet_marker)

        dx_bifurc = ufl.Measure("dx", mesh_3d1d.lm_mesh)

        # Use a constant scalar for h_e inside the 0D term to avoid compiling 1D variables on a 0D domain
        # We calculate the physical cell size (assuming uniform intervals in the submesh)
        coords = edge.geometry.x
        h_e_val = float(np.linalg.norm(coords[1] - coords[0]))

        h_e = ufl.CellDiameter(edge)
        h_e_avg = (h_e("+") + h_e("-"))/2
        n_e = ufl.FacetNormal(edge)

        # DG Form - Diagonal blocks
        a[i+1][i+1] += ufl.inner(ufl.grad(u_1d[i]), ufl.grad(w_1d[i])) * dx_edge
        a[i+1][i+1] -= ufl.inner(ufl.avg(ufl.grad(u_1d[i])), ufl.jump(w_1d[i], n_e)) * dS_edge
        a[i+1][i+1] -= ufl.inner(ufl.avg(ufl.grad(w_1d[i])), ufl.jump(u_1d[i], n_e)) * dS_edge
        a[i+1][i+1] += (sigma_e / h_e_avg) * ufl.inner(ufl.jump(u_1d[i], n_e), ufl.jump(w_1d[i], n_e)) * dS_edge

        # coupling terms
        restriction = Circle(edge, radius, q_degree)

        q_el = basix.ufl.quadrature_element(edge.basix_cell(), value_shape=(), degree=q_degree)
        restriction_space = dolfinx.fem.functionspace(edge, q_el)

        avg_u = Average(u_3d, restriction, restriction_space)
        avg_w = Average(w_3d, restriction, restriction_space)

        a[0][0] += perimeter * xi * ufl.inner(avg_u, avg_w) * dx_edge
        a[i+1][0] -= perimeter * xi * ufl.inner(u_1d[i], avg_w) * dx_edge
        a[0][i+1] -= perimeter * xi * ufl.inner(w_1d[i], avg_u) * dx_edge
        a[i+1][i+1] += perimeter * xi * ufl.inner(u_1d[i], w_1d[i]) * dx_edge

        # Bifurcation node coupling
        a[i+1][i+1] += ufl.inner(ufl.grad(u_1d[i]), ufl.outer(w_1d[i], n_e)) * ds_edge(bifurc_marker)
        a[i+1][i+1] += ufl.inner(ufl.grad(w_1d[i]), ufl.outer(u_1d[i], n_e)) * ds_edge(bifurc_marker)
        a[i+1][i+1] += (sigma_bifurc / h_e) * ufl.inner(u_1d[i], w_1d[i]) * ds_edge(bifurc_marker)

        a[-1][i+1] += ufl.inner(ufl.grad(u_1d[i]), ufl.outer(w_lm, n_e)) * ds_edge(bifurc_marker)
        a[i+1][-1] += ufl.inner(ufl.grad(w_1d[i]), ufl.outer(u_lm, n_e)) * ds_edge(bifurc_marker)

        a[-1][i+1] += (sigma_bifurc / h_e) * ufl.inner(u_1d[i], w_lm) * ds_edge(bifurc_marker)
        a[i+1][-1] += (sigma_bifurc / h_e) * ufl.inner(w_1d[i], u_lm) * ds_edge(bifurc_marker)

        # Use h_e_val (the scalar float) here instead of the UFL object h_e
        a[-1][-1] += (sigma_bifurc / h_e_val) * u_lm * w_lm * dx_bifurc

        # RHS Form
        L[i+1] += ufl.inner(w_1d[i], f_edges[i]) * dx_edge
        #L[i+1] -= ufl.inner(ufl.grad(self._w_e[i]), ufl.outer(self._u_ex_edges[i], n_e)) * ds_edge(self._boundary_marker)
        #L[i+1] += (self._sigma_edge / h_e) * ufl.inner(self._u_ex_edges[i], self._w_e[i]) * ds_edge(self._boundary_marker)

    entity_maps = mesh_3d1d.edge_entity_maps + [mesh_3d1d.lm_map] + mesh_3d1d.lm_to_edge_maps

    for i, ai in enumerate(a):
        for j, aij in enumerate(ai):
            if isinstance(aij, ufl.ZeroBaseForm):
                a[i][j] = None

    # create linear system
    A_blocks = []
    for i, row in enumerate(a):
        A_row = []
        for j, form in enumerate(row):
            if form is not None:
                #print(f"{i}, {j}")
                A_ij = assemble_matrix(form, entity_maps=entity_maps)
                A_ij.assemble()
                A_row.append(A_ij)
                #print("success")
            else:
                #print(f"{i}, {j}")
                A_row.append(None)
                #print("success")
        A_blocks.append(A_row)

    A_mat = PETSc.Mat().createNest(A_blocks, comm=comm)
    A_mat.assemble()

    # Create a list of the actual DOLFINx function spaces to subscript properly
    dolfinx_spaces = [fs_3d] + fs_edges + [fs_lm]

    L_blocks = []
    for i, form in enumerate(L):
        if not isinstance(form, ufl.ZeroBaseForm):
            L_j = create_vector(form, entity_maps=entity_maps)
            assemble_vector(form, b=L_j, entity_maps=entity_maps)
            L_j.assemble()
            L_blocks.append(L_j)
        else:
            # Directly instantiate a Function on the space and get its PETSc vector via .x.petsc_vec
            space = dolfinx_spaces[i]
            zero_func = dolfinx.fem.Function(space)
            L_j = zero_func.x.petsc_vec
            L_blocks.append(L_j)

    b_vec = PETSc.Vec().createNest(L_blocks, comm=comm)
    b_vec.assemblyBegin()
    b_vec.assemblyEnd()

    X_blocks = []
    for i, form in enumerate(L):
        if not isinstance(form, ufl.ZeroBaseForm):
            X_j = create_vector(form, entity_maps=entity_maps)
            X_blocks.append(X_j)
        else:
            space = dolfinx_spaces[i]
            zero_func = dolfinx.fem.Function(space)
            X_j = zero_func.x.petsc_vec
            X_blocks.append(X_j)

    x_vec = PETSc.Vec().createNest(X_blocks, comm=comm)
    x_vec.assemblyBegin()
    x_vec.assemblyEnd()

    # create solver
    solver = PETSc.KSP().create(comm=comm)
    solver.setOperators(A_mat)
    solver.setOptionsPrefix("graph_poisson_")

    # set options
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }

    opts = PETSc.Options()
    opts.prefixPush(solver.getOptionsPrefix())
    for k, v in petsc_options.items():
        opts.setValue(k, v)
    solver.setFromOptions()
    opts.prefixPop()

    # solve system
    solver.solve(b_vec, x_vec)

    # assign functions to solution
    u_h = []
    u_h.append(dolfinx.fem.Function(fs_3d, name="u_h_3d"))
    for i, space in enumerate(fs_edges):
        u_h.append(dolfinx.fem.Function(space, name=f"u_h_{i}"))
    u_h.append(dolfinx.fem.Function(fs_lm, name=f"u_h_lm"))

    dolfinx.la.petsc._ghost_update(
        x_vec,
        insert_mode=PETSc.InsertMode.ADD,
        scatter_mode=PETSc.ScatterMode.FORWARD,
    )

    dolfinx.fem.petsc.assign(x_vec, u_h)

    return u_h, mesh_3d1d, dolfinx_spaces

if __name__ == "__main__":

    # jump penalties
    sigma_omega = 20
    sigma_e = 20
    sigma_bifurc = 20

    # domain data
    xi = 1
    radius = 0.05
    area = ufl.pi * radius ** 2
    perimeter = 2 * ufl.pi * radius

    q_degree = 20

    u_hfine, mesh_3d1d, f_spaces = solve_3d1d(32)

    # get fine grid solution and function spaces
    u_3dfine = u_hfine[0]
    u_1dfine = u_hfine[1:-1]
    u_lmfine = u_hfine[-1]

    fs_3dfine = f_spaces[0]
    fs_1dfine = f_spaces[1:-1]
    fs_lmfine = f_spaces[-1]

    # get cell maps for interpolation data
    cell_map3d = fs_3dfine.mesh.topology.index_map(fs_3dfine.mesh.topology.dim)
    num_cells3d = cell_map3d.size_local + cell_map3d.num_ghosts
    cells3d = np.arange(num_cells3d, dtype=np.int32)

    cells1d = []

    for i, fs in enumerate(fs_1dfine):
        cell_map1d = fs.mesh.topology.index_map(fs.mesh.topology.dim)
        num_cells1d = cell_map1d.size_local + cell_map1d.num_ghosts
        cells1d.append(np.arange(num_cells1d, dtype=np.int32))

    cell_maplm = fs_lmfine.mesh.topology.index_map(fs_lmfine.mesh.topology.dim)
    num_cellslm = cell_maplm.size_local + cell_maplm.num_ghosts
    cellslm = np.arange(num_cellslm, dtype=np.int32)

    Ns = [2, 4, 8]

    for N in Ns:

        # solve for size N
        u_h, mesh_3d1d_h, fs_h = solve_3d1d(N)

        # get solutions and function spaces
        u_h3d = u_h[0]
        u_h1d = u_h[1:-1]
        u_hlm = u_h[-1]

        fs_3d_h = fs_h[0]
        fs_1d_h = fs_h[1:-1]
        fs_lm_h = fs_h[-1]

        # create interpolation data
        interpolation_data3d = dolfinx.fem.create_interpolation_data(fs_3dfine, fs_3d_h, cells3d)
        
        interpolation_data1d = []
        for fs, fs_fine, cells in zip(fs_1d_h, fs_1dfine, cells1d):
            interps = dolfinx.fem.create_interpolation_data(fs_fine, fs, cells)
            interpolation_data1d.append(interps)

        # create and interpolation solution functions
        u_h3dfine = dolfinx.fem.Function(fs_3dfine)
        u_h3dfine.interpolate_nonmatching(u_h3d, cells3d, interpolation_data3d)

        u_h1dfine = []
        for i in range(len(u_h1d)):
            u_he = dolfinx.fem.Function(fs_1dfine[i])
            u_he.interpolate_nonmatching(u_h1d[i], cells1d[i], interpolation_data1d[i])
            u_h1dfine.append(u_he)

        u_hlmfine = dolfinx.fem.Function(fs_lmfine)
        u_hlmfine.interpolate(u_hlm)

        # error on 3d mesh
        dx3d = ufl.Measure("dx", domain=mesh_3d1d.mesh3D)
        dS3d = ufl.Measure("dS", domain=mesh_3d1d.mesh3D)
        ds3d = ufl.Measure("ds", domain=mesh_3d1d.mesh3D)

        h_omega = ufl.CellDiameter(mesh_3d1d.mesh3D)
        h_omega_avg = (h_omega("+") + h_omega("-")) / 2
        #n_omega = ufl.FacetNormal(mesh_3d1d.mesh3D)

        Omega_H1err = ufl.inner(ufl.grad(u_h3dfine - u_3dfine), ufl.grad(u_h3dfine - u_3dfine)) * dx3d
        Omega_err = assemble_scalar(Omega_H1err, op=MPI.SUM)
        Omega_jumperr = (sigma_omega / h_omega_avg) * ufl.inner(ufl.jump(u_h3dfine - u_3dfine), ufl.jump(u_h3dfine - u_3dfine)) * dS3d
        Omega_err += assemble_scalar(Omega_jumperr, op=MPI.SUM)
        Omega_outer_jumperr = (sigma_omega / h_omega) * ufl.inner(u_h3dfine - u_3dfine, u_h3dfine - u_3dfine) * ds3d
        Omega_err += assemble_scalar(Omega_outer_jumperr, op=MPI.SUM)

        # error on 1d meshes

        G_err = 0
        couple_err = 0
        lm_err = 0

        for i in range(len(fs_1d_h)):

            edge = mesh_3d1d.edge_meshes[i]

            dx_e = ufl.Measure("dx", domain=mesh_3d1d.edge_meshes[i])
            dS_e = ufl.Measure("dS", domain=mesh_3d1d.edge_meshes[i])
            ds_e = ufl.Measure("ds", domain=mesh_3d1d.edge_meshes[i])

            h_e = ufl.CellDiameter(mesh_3d1d.edge_meshes[i])
            h_e_avg = (h_e("+") + h_e("-")) / 2
            #n_omega = ufl.FacetNormal(mesh_3d1d.mesh3D)

            J = ufl.Jacobian(mesh_3d1d.edge_meshes[i])
            tangent = J[:,0]
            norm_tangent = tangent / ufl.sqrt(ufl.dot(tangent, tangent))

            du_e_h = ufl.dot(ufl.grad(u_h1dfine[i]), norm_tangent)
            du_e_fine = ufl.dot(ufl.grad(u_1dfine[i]), norm_tangent)

            edge_H1err = area * ufl.inner(du_e_h - du_e_fine, du_e_h - du_e_fine) * dx_e
            G_err += assemble_scalar(edge_H1err, op=MPI.SUM)
            edge_jumperr = area * (sigma_e / h_e_avg) * ufl.inner(ufl.jump(u_h1dfine[i] - u_1dfine[i]), ufl.jump(u_h1dfine[i] - u_1dfine[i])) * dS_e
            G_err += assemble_scalar(edge_jumperr, op=MPI.SUM)

            # coupling error

            restriction = Circle(mesh_3d1d.edge_meshes[i], radius, q_degree)

            q_el = basix.ufl.quadrature_element(edge.basix_cell(), value_shape=(), degree=q_degree)
            restriction_space = dolfinx.fem.functionspace(edge, q_el)

            avg_ufine = Average(u_3dfine, restriction, restriction_space)
            avg_uhfine = Average(u_h3dfine, restriction, restriction_space)

            edge_couple_err = perimeter * xi * ufl.inner((avg_ufine - avg_uhfine) - (u_1dfine[i] - u_h1dfine[i]), (avg_ufine - avg_uhfine) - (u_1dfine[i] - u_h1dfine[i])) * dx_e
            couple_err += assemble_scalar(edge_couple_err, op=MPI.SUM)

            # lm error

            h_cell = edge.h(1, np.array([0]))[0]
            lm_err += (sigma_bifurc / h_cell) * ((u_h1dfine[i].x.array[-1] - u_1dfine[i].x.array[-1])-(u_hlmfine.x.array[0] - u_lmfine.x.array[0])) ** 2

        DG_err = np.sqrt(Omega_err + G_err + couple_err + lm_err)

        print(f"N = {N}, err = {DG_err}")