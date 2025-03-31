import numpy as np
import cupy as cp
import cupyx
from cupyx.scipy.sparse import coo_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import gmres
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import json
import os
import time

print("CuPy Version:", cp.__version__)
print(cp.show_config())

###############################################################################
# 1. Data Classes
###############################################################################
class ReservoirGrid:
    def __init__(self, nx, ny, dx, dy, phi, k_matrix, k_frac_map, ct, mu, p_init=3.5e7):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.phi = phi
        self.k_matrix = k_matrix
        self.ct = ct
        self.mu = mu
        self.k = np.ones((nx, ny), dtype=float) * k_matrix
        for (ix, iy), kf in k_frac_map.items():
            self.k[ix, iy] = kf
        self.pressure = np.ones((nx, ny), dtype=float) * p_init
        self.thickness = 1.0
        self.cell_volume = dx * dy * self.thickness
        self.bc_dirichlet = {}

class Fracture:
    def __init__(self, x_start, y_start, length_cells, frac_transmissibility=1e-12, p_frac=1e5):
        self.x_start = x_start
        self.y_start = y_start
        self.length_cells = length_cells
        self.frac_trans = frac_transmissibility
        self.pressure = p_frac

###############################################################################
# 2. GPU-enabled Build Matrices
###############################################################################
def build_transmissibility_matrix_gpu(grid):
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    thickness = grid.thickness
    mu = grid.mu
    row_inds, col_inds, vals = [], [], []
    def idx(i, j):
        return i + j * nx
    for j in range(ny):
        for i in range(nx):
            I = idx(i, j)
            diag = 0.0
            k_ij = grid.k[i, j]
            if i > 0:
                k_left = grid.k[i-1, j]
                k_face = (2 * k_ij * k_left) / (k_ij + k_left + 1e-30)
                T_x = k_face * thickness / mu * (dy/dx)
                row_inds.append(I)
                col_inds.append(idx(i-1, j))
                vals.append(-T_x)
                diag += T_x
            if i < nx-1:
                k_right = grid.k[i+1, j]
                k_face = (2 * k_ij * k_right) / (k_ij + k_right + 1e-30)
                T_x = k_face * thickness / mu * (dy/dx)
                row_inds.append(I)
                col_inds.append(idx(i+1, j))
                vals.append(-T_x)
                diag += T_x
            if j > 0:
                k_down = grid.k[i, j-1]
                k_face = (2 * k_ij * k_down) / (k_ij + k_down + 1e-30)
                T_y = k_face * thickness / mu * (dx/dy)
                row_inds.append(I)
                col_inds.append(idx(i, j-1))
                vals.append(-T_y)
                diag += T_y
            if j < ny-1:
                k_up = grid.k[i, j+1]
                k_face = (2 * k_ij * k_up) / (k_ij + k_up + 1e-30)
                T_y = k_face * thickness / mu * (dx/dy)
                row_inds.append(I)
                col_inds.append(idx(i, j+1))
                vals.append(-T_y)
                diag += T_y
            row_inds.append(I)
            col_inds.append(I)
            vals.append(diag)
    N = nx * ny
    row_cupy = cp.array(row_inds, dtype=cp.int32)
    col_cupy = cp.array(col_inds, dtype=cp.int32)
    vals_cupy = cp.array(vals, dtype=cp.float64)
    T_coo_gpu = coo_matrix((vals_cupy, (row_cupy, col_cupy)), shape=(N, N))
    T_csr_gpu = T_coo_gpu.tocsr()
    return T_csr_gpu

def build_mass_matrix_gpu(grid, dt):
    nx, ny = grid.nx, grid.ny
    N = nx * ny
    accum_val = grid.phi * grid.ct * grid.cell_volume / dt
    diagvals = cp.full(N, accum_val, dtype=cp.float64)
    M_gpu = cupyx.scipy.sparse.diags(diagvals, 0, shape=(N, N), format='csr')
    return M_gpu

###############################################################################
# 3. Fracture Source Terms and Pressure Solve (GPU)
###############################################################################
def compute_fracture_source_terms(grid, fractures, prod_rate=-0.001):
    nx, ny = grid.nx, grid.ny
    Q = np.zeros(nx*ny, dtype=float)
    def idx(i, j):
        return i + j * nx
    for frac in fractures:
        x0 = frac.x_start
        y0 = frac.y_start
        if 0 <= x0 < nx and 0 <= y0 < ny:
            cell_idx = idx(x0, y0)
            p_cell = grid.pressure[x0, y0]
            p_frac = frac.pressure
            Q[cell_idx] += frac.frac_trans * (p_frac - p_cell) + prod_rate
    return Q

def solve_pressure_implicit_gpu(grid, T_gpu, dt, fractures, prod_rate):
    nx, ny = grid.nx, grid.ny
    M_gpu = build_mass_matrix_gpu(grid, dt)
    P_old_gpu = cp.array(grid.pressure.ravel(), dtype=cp.float64)
    Q_cpu = compute_fracture_source_terms(grid, fractures, prod_rate)
    Q_gpu = cp.array(Q_cpu, dtype=cp.float64)
    A_gpu = M_gpu + T_gpu
    b_gpu = M_gpu.dot(P_old_gpu) + Q_gpu
    def idx(i, j):
        return i + j * nx
    for (i_bc, j_bc), p_bc in grid.bc_dirichlet.items():
        bc_index = idx(i_bc, j_bc)
        start = A_gpu.indptr[bc_index]
        end = A_gpu.indptr[bc_index+1]
        A_gpu.data[start:end] = 0.0
        A_gpu[bc_index, bc_index] = 1.0
        b_gpu[bc_index] = p_bc
    sol_gpu, info = gmres(A_gpu, b_gpu, x0=P_old_gpu, tol=1e-8, maxiter=10000)
    if info != 0:
        print(f"Warning: GMRES did not fully converge, info = {info}")
    P_new_cpu = cp.asnumpy(sol_gpu)
    grid.pressure = P_new_cpu.reshape((nx, ny))

###############################################################################
# 4. Main Simulation Function with Section Timings and Progress Bar (GPU)
###############################################################################
def run_simulation(
    nx=100, ny=100, dx=10.0, dy=10.0,
    phi=0.2, k_matrix=1e-15,
    frac_cols=None, ct=1e-9, mu=1e-3, p_init=3.5e7, well_p=1e5,
    n_steps=30, dt=300.0, prod_rate=-0.001,
    frac_length_cells=10, frac_length_list=None, frac_bc_fraction=0.8
):
    perf = {}
    t0 = time.time()
    
    # Grid and BC Initialization
    t_grid_start = time.time()
    if frac_cols is None:
        frac_cols = [25]
    if frac_length_list is None:
        frac_length_list = [frac_length_cells] * len(frac_cols)
    k_frac_map = {}
    for col in frac_cols:
        for row in range(ny):
            k_frac_map[(col, row)] = 1e-12
    grid = ReservoirGrid(nx, ny, dx, dy, phi, k_matrix, k_frac_map, ct, mu, p_init)
    for j in range(ny):
        grid.bc_dirichlet[(0, j)] = p_init
        grid.bc_dirichlet[(nx-1, j)] = p_init
    for i in range(nx):
        grid.bc_dirichlet[(i, 0)] = p_init
        grid.bc_dirichlet[(i, ny-1)] = p_init
    t_grid_end = time.time()
    perf['grid_initialization_time_gpu'] = t_grid_end - t_grid_start

    # Set forced BC for fractures
    t_bc_start = time.time()
    mid_y = ny // 2
    for col, L in zip(frac_cols, frac_length_list):
        buffer = int((1 - frac_bc_fraction) * L / 2)
        j_min = max(0, mid_y - L//2 + buffer)
        j_max = min(ny-1, mid_y + L//2 - buffer)
        for j in range(j_min, j_max+1):
            grid.bc_dirichlet[(col, j)] = well_p
    t_bc_end = time.time()
    perf['bc_setting_time_gpu'] = t_bc_end - t_bc_start

    # Build Transmissibility Matrix
    t_T_start = time.time()
    T_gpu = build_transmissibility_matrix_gpu(grid)
    t_T_end = time.time()
    perf['build_transmissibility_matrix_gpu_time'] = t_T_end - t_T_start

    # Create Fracture Objects
    t_fracture_start = time.time()
    fractures = []
    alpha = 1e4
    for col, L in zip(frac_cols, frac_length_list):
        p_eff = max(0, well_p - alpha * L)
        fractures.append(Fracture(x_start=col, y_start=mid_y, length_cells=L,
                                  frac_transmissibility=1e-7, p_frac=p_eff))
    t_fracture_end = time.time()
    perf['fracture_creation_time_gpu'] = t_fracture_end - t_fracture_start

    # Time-stepping loop with progress bar
    t_sim_start = time.time()
    step_times = []
    times_list = []
    for step in range(n_steps):
        t_step0 = time.time()
        solve_pressure_implicit_gpu(grid, T_gpu, dt, fractures, prod_rate)
        t_step1 = time.time()
        step_times.append(t_step1 - t_step0)
        times_list.append((step+1) * dt)
        progress = (step + 1) / n_steps
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '=' * filled + '-' * (bar_length - filled)
        print(f"\rTime step {step+1}/{n_steps} [{bar}] {progress*100:5.1f}% completed", end='')
    t_sim_end = time.time()
    print("\nGPU Simulation completed.")
    perf['simulation_loop_total_time_gpu'] = t_sim_end - t_sim_start
    perf['average_time_per_timestep_gpu'] = np.mean(step_times)
    perf['overall_run_time_gpu'] = time.time() - t0

    time_complexity = {
        "grid_initialization": "O(nx*ny)",
        "build_transmissibility_matrix_gpu": "O(nx*ny)",
        "build_mass_matrix_gpu": "O(nx*ny)",
        "compute_fracture_source_terms": "O(number of fractures)",
        "solve_pressure_implicit_gpu": "GMRES: worst-case O(iterations * nnz)",
        "simulation_loop": f"{n_steps} iterations"
    }
    return grid, fractures, times_list, None, perf, time_complexity

###############################################################################
# 5. Flask App: Endpoints and Output Files (GPU)
###############################################################################
app = Flask(__name__)
CORS(app)
RESULT_FILENAME = "simulation_results.json"
PERFORMANCE_LOG_FILENAME = "performance_log.json"

def flatten_2d(arr):
    return [float(x) for x in arr.ravel().tolist()]

@app.route('/simulation_data')
def serve_simulation_data():
    if os.path.exists(RESULT_FILENAME):
        return send_from_directory('.', RESULT_FILENAME)
    else:
        return "Simulation data not available. Please run the simulation first.", 404

@app.route('/run', methods=['POST'])
def run_simulation_endpoint():
    print("Running Simulation (GPU version)...")
    nx, ny, dx, dy = 150, 150, 10.0, 10.0
    frac_cols = [5, 15, 20, 25, 30, 45]
    frac_length_list = [5, 15, 7, 12, 4, 35]
    grid, fracs, times, pressures, perf, complexity = run_simulation(
        nx=nx, ny=ny, dx=dx, dy=dy, phi=0.2, k_matrix=1e-15,
        frac_cols=frac_cols, ct=1e-9, mu=1e-3, p_init=35e6, well_p=25e6,
        n_steps=300, dt=300.0, prod_rate=-0.001,
        frac_length_cells=10, frac_length_list=frac_length_list
    )
    
    # Prepare simulation results used to initialize the viewer.
    simulation_data = {
        "nx": nx, "ny": ny, "dx": dx, "dy": dy,
        "phi": 0.2, "k_matrix": 0.001e-15, "ct": 1e-9, "mu": 1e-3,
        "p_init": 3.5e7, "well_p": 1e5,
        "timeSteps": times,
        "notes": "GPU-based simulation; timings are in seconds."
    }
    
    # Write the simulation results file.
    with open(RESULT_FILENAME, "w") as f:
        json.dump(simulation_data, f)
    print("\nSimulation results saved to", RESULT_FILENAME)
    
    # Write the separate performance log file.
    with open(PERFORMANCE_LOG_FILENAME, "w") as f:
        json.dump(perf, f)
    print("Performance log saved to", PERFORMANCE_LOG_FILENAME)
    
    return jsonify({"message": "Simulation run (GPU version) completed and data saved."})

if __name__ == '__main__':
    print("Flask running on http://localhost:5000 (GPU version)")
    app.run(port=5000, debug=True)
