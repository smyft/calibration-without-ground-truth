import numpy as np
import cvxpy as cp
import scipy.sparse as sp
import time
from .diagnostics import ProjectionDiagnostics

def solve_minimax_sparse_vectorized(P_r1r2, r1_j, r1_m, r2_j, r2_m, K, debug=False,
                                    diagnostics_path=None):
    diagnostics = ProjectionDiagnostics(diagnostics_path) if diagnostics_path is not None else None
    SCALE_FACTOR = 1e3
    P_r1r2 = np.asarray(P_r1r2)
    n_r1, n_r2 = P_r1r2.shape
    
    P_scaled_vals = P_r1r2 * SCALE_FACTOR

    r1_j = np.asarray(r1_j, dtype=int)
    r1_m = np.asarray(r1_m, dtype=float)
    r2_j = np.asarray(r2_j, dtype=int)
    r2_m = np.asarray(r2_m, dtype=float)

    # 1. Coordinate format extraction of non-zero P(r1, r2)
    # This avoids creating variables for zero-probability events
    rows, cols = np.nonzero(P_scaled_vals > 1e-9)
    vals = P_scaled_vals[rows, cols]
    nnz = len(vals)

    if debug:
        print(f"Starting Vectorized Solver. Grid: {n_r1}x{n_r2}. NNZ: {nnz}")

    # 2. Define Variables
    # pi_flat represents pi(theta, r1, r2) only for valid (r1,r2) pairs
    # Shape: (K, nnz)
    pi_flat = cp.Variable((K, nnz), nonneg=True)
    
    # 3. Aggregation Logic using Sparse Matrix Multiplication
    # We need u[theta, a] = sum_{b} pi[theta, a, b]
    # We need v[theta, b] = sum_{a} pi[theta, a, b]
    
    # Construct selection matrices.
    # S1 maps flat_index -> row_index (r1)
    # S2 maps flat_index -> col_index (r2)
    # Valid indices range from 0 to nnz-1. 
    # S1 has shape (nnz, n_r1). multiplying (K, nnz) @ (nnz, n_r1) -> (K, n_r1)
    
    # Data is just 1s because we are summing. 
    ones_data = np.ones(nnz)
    range_data = np.arange(nnz)
    # S1: rows are flat_indices, cols are r1 indices
    S1 = sp.coo_matrix((ones_data, (range_data, rows)), shape=(nnz, n_r1))
    # S2: rows are flat_indices, cols are r2 indices
    S2 = sp.coo_matrix((ones_data, (range_data, cols)), shape=(nnz, n_r2))

    u = pi_flat @ S1  # Shape: (K, n_r1)
    v = pi_flat @ S2  # Shape: (K, n_r2)
    
    P1 = P_scaled_vals.sum(axis=1)
    P2 = P_scaled_vals.sum(axis=0)

    constraints = []
    
    # Constraint 1: Consistency with P(r1, r2) values
    # sum(pi_flat, axis=0) should match the probability mass for that (r1,r2) pair
    constraints.append(cp.sum(pi_flat, axis=0) == vals)

    # Constraint 2: r2 calibration (Acting on v)
    # Optimization: Only iterate over r2 indices that actually appeared in the non-zeros
    active_b_mask = P2 > 1e-9
    active_b_indices = np.where(active_b_mask)[0] # e.g. [0, 1, 4...]
    
    v_active = v[:, active_b_indices]   # shape (K, num_active_b)
    r2_j_active = r2_j[active_b_indices]
    r2_m_active = r2_m[active_b_indices]
    P2_active   = P2[active_b_indices]
    
    # 2.1 equation: v[predicted_class, col] == m * P
    col_selectors = np.arange(len(active_b_indices))
    constraints.append(
        v_active[r2_j_active, col_selectors] == r2_m_active * P2_active
    )
    
    # 2.2 Argmax: v[predicted, col] >= v[any_t, col]
    # choose target v，reshape to (1, N) for broadcast
    v_target_vals = v_active[r2_j_active, col_selectors] # Shape (N,)
    v_target_vals = cp.reshape(v_target_vals, (1, len(active_b_indices)))
    
    # constraint: every row of v_active <= target
    constraints.append(v_active <= v_target_vals)


    # --- Constraint 3 & Objective ---
    active_a_mask = P1 > 1e-9
    active_a_indices = np.where(active_a_mask)[0]
    
    u_active = u[:, active_a_indices] # Shape (K, num_active_a)
    P1_active = P1[active_a_indices]
    r1_j_active = r1_j[active_a_indices]
    r1_m_active = r1_m[active_a_indices]
    
    # variables t_var can be only defined over active r1
    n_active_a = len(active_a_indices)
    t_var = cp.Variable(n_active_a, nonneg=True)
    
    # 3.1 constraint: u[t, a]^2 <= P1[a] * t_var[a]
    # RHS: P1[a] * t_var[a] shape (N,)
    # LHS: u_active shape (K, N)
    # broadcast RHS to (1, N)
    rhs_vec = cp.multiply(P1_active, t_var) # Shape (N,)
    rhs_broadcast = cp.reshape(rhs_vec, (1, n_active_a)) # Shape (1, N)
    
    # constraint: every row of LHS  <= RHS
    constraints.append(cp.square(u_active) <= rhs_broadcast)
    
    # 3.2 Objective function
    # sum( t_var - 2 * m * u[predicted] )
    
    term1 = cp.sum(t_var)
    
    # 第二项: sum( 2 * m * u_selected )
    col_selectors_a = np.arange(n_active_a)
    u_selected = u_active[r1_j_active, col_selectors_a] # Shape (N,)
    
    term2 = 2.0 * cp.sum(cp.multiply(r1_m_active, u_selected))
    
    # constant CP1*m^2 can be omitted
    objective = cp.Minimize(term1 - term2)
    
    prob = cp.Problem(objective, constraints)

    def record_diagnostics(stats, mapping, error=None):
        if diagnostics is not None:
            diagnostics.finish(
                problem=prob, solver_stats=stats, pi=pi_flat, t=t_var, u=u, v=v,
                P=P_r1r2, P1=P1, P2=P2, rows=rows, cols=cols,
                r1_j=r1_j, r1_m=r1_m, r2_j=r2_j, r2_m=r2_m,
                active_a=active_a_indices, active_b=active_b_indices,
                scale=SCALE_FACTOR, mapping=mapping, debug=debug, error=error,
            )

    t_start = time.time()
    try:
        if debug: print("Solver started...")
        prob.solve(solver=cp.CLARABEL, verbose=debug)
    except cp.error.SolverError as e:
        print(f"Solver Error: {e}")
        solver_stats = {
            'n_r1': n_r1, 'n_r2': n_r2, 'nnz': nnz, 'n_vars': K * nnz,
            'solver_time_s': time.time() - t_start, 'solver_status': 'error',
        }
        record_diagnostics(solver_stats, None, str(e))
        return solver_stats, None
    solve_time = time.time() - t_start

    solver_stats = {
        'n_r1': n_r1,
        'n_r2': n_r2,
        'nnz': nnz,
        'n_vars': K * nnz,
        'solver_time_s': solve_time,
        'solver_status': prob.status,
    }

    if debug:
        print(f"Solved. Status: {prob.status}. Time: {solve_time:.2f}s. "
              f"Grid: {n_r1}x{n_r2}. NNZ: {nnz}. Vars: {K * nnz}.")

    # =========================================================================
    # RECONSTRUCT OUTPUT
    # =========================================================================
    if pi_flat.value is None:
        record_diagnostics(solver_stats, [])
        return solver_stats, []

    # Get u values directly from the expression value
    u_opt_val = u.value # (K, n_r1)
    
    # Compute f_mapping using u (marginal over r2)
    f_mapping = []
    for a in range(n_r1):
        if P1[a] > 1e-9:
            # Reconstruct conditional approach q(theta|r1)
            gamma_a = u_opt_val[:, a] / P1[a]
            if diagnostics is not None:
                diagnostics.capture_q(a, "before_clipping", gamma_a)
            # Handle numerical noise
            gamma_a = np.maximum(gamma_a, 0)
            if diagnostics is not None:
                diagnostics.capture_q(a, "after_clipping", gamma_a)
            gamma_a /= (np.sum(gamma_a) + 1e-12) # normalize just in case
            if diagnostics is not None:
                diagnostics.capture_q(a, "after_normalization", gamma_a)
            
            j_star = np.argmax(gamma_a)
            m_star = gamma_a[j_star]
            f_mapping.append((j_star, m_star))
        else:
            f_mapping.append((-1, 0.0))

    record_diagnostics(solver_stats, f_mapping)
    return solver_stats, f_mapping
