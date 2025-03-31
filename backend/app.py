import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import json
import os

###############################################################################
# 1. Data Classes
###############################################################################
class ReservoirGrid:
    def __init__(self, nx, ny, dx, dy,
                 phi, k_matrix, k_frac_map,
                 ct, mu, p_init=3.5e7):
        """
        2D grid: nx x ny, with each cell dx by dy (m).
        p_init: initial reservoir pressure in Pa.
        k_matrix: base permeability in m^2.
        k_frac_map: dict {(ix, iy): k_frac} for fracture cells.
        ct: total compressibility in 1/Pa.
        mu: fluid viscosity in Pa.s.
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.phi = phi
        self.k_matrix = k_matrix
        self.ct = ct
        self.mu = mu
        # Build permeability field; override with k_frac where specified.
        self.k = np.ones((nx, ny), dtype=float) * k_matrix
        for (ix, iy), kf in k_frac_map.items():
            self.k[ix, iy] = kf
        # Uniform initial pressure.
        self.pressure = np.ones((nx, ny), dtype=float) * p_init
        self.thickness = 1.0
        self.cell_volume = dx * dy * self.thickness
        self.bc_dirichlet = {}

class Fracture:
    def __init__(self, x_start, y_start, length_cells,
                 frac_transmissibility=1e-12,
                 p_frac=1e5):
        """
        Represents a vertical fracture column.
        """
        self.x_start = x_start
        self.y_start = y_start
        self.length_cells = length_cells
        self.frac_trans = frac_transmissibility
        self.pressure = p_frac

###############################################################################
# 2. Building Matrices
###############################################################################
def build_transmissibility_matrix(grid):
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    thickness = grid.thickness
    mu = grid.mu
    row_inds = []
    col_inds = []
    vals = []
    def idx(i, j):
        return i + j * nx
    for j in range(ny):
        for i in range(nx):
            I = idx(i, j)
            diag = 0.0
            k_ij = grid.k[i, j]
            # left neighbor
            if i > 0:
                k_left = grid.k[i-1, j]
                k_face = (2*k_ij*k_left) / (k_ij + k_left + 1e-30)
                T_x = k_face * thickness / mu * (dy/dx)
                row_inds.append(I)
                col_inds.append(idx(i-1, j))
                vals.append(-T_x)
                diag += T_x
            # right neighbor
            if i < nx-1:
                k_right = grid.k[i+1, j]
                k_face = (2*k_ij*k_right) / (k_ij + k_right + 1e-30)
                T_x = k_face * thickness / mu * (dy/dx)
                row_inds.append(I)
                col_inds.append(idx(i+1, j))
                vals.append(-T_x)
                diag += T_x
            # down neighbor
            if j > 0:
                k_down = grid.k[i, j-1]
                k_face = (2*k_ij*k_down) / (k_ij + k_down + 1e-30)
                T_y = k_face * thickness / mu * (dx/dy)
                row_inds.append(I)
                col_inds.append(idx(i, j-1))
                vals.append(-T_y)
                diag += T_y
            # up neighbor
            if j < ny-1:
                k_up = grid.k[i, j+1]
                k_face = (2*k_ij*k_up) / (k_ij + k_up + 1e-30)
                T_y = k_face * thickness / mu * (dx/dy)
                row_inds.append(I)
                col_inds.append(idx(i, j+1))
                vals.append(-T_y)
                diag += T_y
            row_inds.append(I)
            col_inds.append(I)
            vals.append(diag)
    N = nx * ny
    T = sp.coo_matrix((vals, (row_inds, col_inds)), shape=(N, N)).tocsr()
    return T

def build_mass_matrix(grid, dt):
    nx, ny = grid.nx, grid.ny
    N = nx * ny
    accumulation_val = grid.phi * grid.ct * grid.cell_volume / dt
    M = sp.diags([accumulation_val]*N, 0, shape=(N, N), format='csr')
    return M

###############################################################################
# 3. Fracture Source Terms and Pressure Solve
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
            Q[cell_idx] += frac.frac_trans * (p_frac - p_cell)
            Q[cell_idx] += prod_rate
    return Q

def solve_pressure_implicit(grid, T, dt, fractures, prod_rate):
    nx, ny = grid.nx, grid.ny
    M = build_mass_matrix(grid, dt)
    P_old = grid.pressure.ravel()
    Q = compute_fracture_source_terms(grid, fractures, prod_rate)
    A = M + T
    b = M.dot(P_old) + Q
    def idx(i, j):
        return i + j * nx
    for (i_bc, j_bc), p_bc in grid.bc_dirichlet.items():
        bc_index = idx(i_bc, j_bc)
        A.data[A.indptr[bc_index]:A.indptr[bc_index+1]] = 0.0
        A[bc_index, bc_index] = 1.0
        b[bc_index] = p_bc
    P_new = spla.spsolve(A, b)
    grid.pressure = P_new.reshape((nx, ny))

###############################################################################
# 4. Main Simulation Function
###############################################################################
def run_simulation(
    nx=100, ny=100, dx=10.0, dy=10.0,
    phi=0.2, k_matrix=1e-15,
    frac_cols=None, ct=1e-9, mu=1e-3, p_init=3.5e7,
    well_p=1e5, n_steps=300, dt=3600.0, prod_rate=-0.001,
    frac_length_cells=10, frac_length_list=None, frac_bc_fraction=0.1
):
    perf = {}
    import time
    t0 = time.time()

    if frac_cols is None:
        frac_cols = [25]
    if frac_length_list is None:
        frac_length_list = [frac_length_cells] * len(frac_cols)
    k_frac_map = {}
    for col in frac_cols:
        for row in range(ny):
            k_frac_map[(col, row)] = 1e-12

    t_grid_start = time.time()
    grid = ReservoirGrid(nx, ny, dx, dy, phi, k_matrix, k_frac_map, ct, mu, p_init)
    for j in range(ny):
        grid.bc_dirichlet[(0, j)] = p_init
        grid.bc_dirichlet[(nx-1, j)] = p_init
    for i in range(nx):
        grid.bc_dirichlet[(i, 0)] = p_init
        grid.bc_dirichlet[(i, ny-1)] = p_init
    t_grid_end = time.time()
    perf['grid_initialization_time'] = t_grid_end - t_grid_start

    t_bc_start = time.time()
    mid_y = ny // 2
    for col, L in zip(frac_cols, frac_length_list):
        buffer = int((1 - frac_bc_fraction) * L / 2)
        j_min = max(0, mid_y - L//2 + buffer)
        j_max = min(ny-1, mid_y + L//2 - buffer)
        for j in range(j_min, j_max+1):
            grid.bc_dirichlet[(col, j)] = well_p
    t_bc_end = time.time()
    perf['bc_setting_time'] = t_bc_end - t_bc_start

    t_T_start = time.time()
    T = build_transmissibility_matrix(grid)
    t_T_end = time.time()
    perf['build_transmissibility_matrix_time'] = t_T_end - t_T_start

    t_fracture_start = time.time()
    fractures = []
    alpha = 1e4
    for col, L in zip(frac_cols, frac_length_list):
        p_eff = max(0, well_p - alpha * L)
        fractures.append(Fracture(x_start=col, y_start=mid_y, length_cells=L,
                                  frac_transmissibility=1e-5, p_frac=p_eff))
    t_fracture_end = time.time()
    perf['fracture_creation_time'] = t_fracture_end - t_fracture_start

    times = []
    pressures_over_time = []
    for step in range(n_steps):
        current_t = (step + 1) * dt
        times.append(current_t)
        solve_pressure_implicit(grid, T, dt, fractures, prod_rate)
        pressures_over_time.append(np.copy(grid.pressure))
        progress = (step + 1) / n_steps
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '=' * filled + '-' * (bar_length - filled)
        print(f"\rTime step {step+1}/{n_steps} [{bar}] {progress*100:5.1f}% completed", end='')
    print("\nCPU Simulation completed.")
    perf['simulation_loop_total_time'] = time.time() - t0
    perf['average_time_per_timestep'] = perf['simulation_loop_total_time'] / n_steps
    perf['overall_run_time'] = time.time() - t0

    return grid, fractures, times, pressures_over_time, perf

###############################################################################
# 5. Flask App: Endpoints and Output Files
###############################################################################
app = Flask(__name__)
CORS(app)

RESULT_FILENAME = os.path.join(app.root_path, "simulation_results.json")
PERFORMANCE_LOG_FILENAME = os.path.join(app.root_path, "performance_log.json")

def flatten_2d(arr):
    return [float(x) for x in arr.ravel().tolist()]

@app.route('/simulation_data')
def serve_simulation_data():
    if os.path.exists(RESULT_FILENAME):
        return send_from_directory(app.root_path, "simulation_results.json")
    else:
        return "Simulation data not available. Please run the simulation first.", 404

@app.route('/run', methods=['POST'])
def run_simulation_endpoint():
    print("Running CPU Simulation....")
    nx = 150
    ny = 110
    dx = 10.0
    dy = 10.0
    frac_cols = [20, 25, 30, 45, 55, 65, 70, 85, 88, 96, 106, 115, 125]
    frac_length_list =  [52, 35, 18, 12, 4, 35, 43, 22, 14, 53, 23, 11, 8, 19]

    grid, fracs, times, p_hist, perf = run_simulation(
        nx=nx, ny=ny, dx=dx, dy=dy,
        phi=0.2, k_matrix=7e-15,
        frac_cols=frac_cols,
        ct=1e-9, mu=1e-3,
        p_init=35e6,
        well_p=25e6,
        n_steps=1000, dt=600.0,
        prod_rate=-0.001,
        frac_length_cells=10,
        frac_length_list=frac_length_list,
        frac_bc_fraction=0.1
    )

    pressures_flat = [flatten_2d(p) for p in p_hist]

    frac_data = []
    for f in fracs:
        frac_data.append({
            "x_start": int(f.x_start),
            "y_start": int(f.y_start),
            "length_cells": int(f.length_cells),
            "frac_trans": float(f.frac_trans),
            "p_frac": float(f.pressure)
        })

    simulation_results = {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "phi": 0.2,
        "k_matrix": 0.001e-15,
        "ct": 1e-9,
        "mu": 1e-3,
        "p_init": 3.5e7,
        "well_p": 1e5,
        "timeSteps": [float(t) for t in times],
        "pressures": pressures_flat,
        "fractures": frac_data,
        "notes": ("Infinite acting edges; production via vertical fracture columns. "
                  "The forced low-pressure region correlates with fracture length.")
    }

    with open(RESULT_FILENAME, "w") as f:
        json.dump(simulation_results, f)
    print("Simulation results saved to", RESULT_FILENAME)

    with open(PERFORMANCE_LOG_FILENAME, "w") as f:
        json.dump(perf, f)
    print("Performance log saved to", PERFORMANCE_LOG_FILENAME)

    return jsonify({"message": "CPU Simulation run completed and data saved."})

if __name__ == '__main__':
    print("Flask running on http://localhost:5000")
    #run_simulation()
    app.run(port=5000, debug=True)
