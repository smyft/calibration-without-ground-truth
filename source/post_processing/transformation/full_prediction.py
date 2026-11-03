import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import warnings

def solve_optimal_transformation(P_q1q2, q1_values, q2_values, K, debug=False, tolerance=1e-4):
    """
    Solves for the optimal transformation f(q1) in a discrete setting using Convex Optimization.
    
    Problem Formulation:
        We look for the "Worst-Case" ground truth distribution mu (represented by X)
        that generates predictors q1 and q2, subject to q2 being calibrated.
        Then f(q1) is derived from this mu.
        
        Optimization:
            min_{X} E_{q1} [ || f(q1) - q1 ||^2 ]
            
        Where:
            f(q1) = E_[q2|q1] [ X_{q1,q2} ]  (The predictor derived from latent truth X)
            
        Constraints:
            1. E_[q1|q2] [ X_{q1,q2} ] = q2  (Constraint: q2 is unbiased/calibrated)
            2. X_{q1,q2} in Simplex          (X must be a valid probability distribution)
        
    Parameters:
    ----------
    P_q1q2 : np.ndarray (N, M)
        Joint distribution P(q1, q2). 
        Rows correspond to q1 indices, columns to q2 indices.
    q1_values : np.ndarray (N, K)
        Possible values for prediction q1. Each row is a probability vector of dim K.
    q2_values : np.ndarray (M, K)
        Possible values for prediction q2.
    K : int
        Size of the world state space (dimension of q vectors).
    debug : bool
        Whether to print solver logs.
        
    Returns:
    ----------
    result_map : dict
        Mapping from q1 index (int) to the optimized prediction (np.array).
        f(q1_values[i]) = result_map[i]
    """
    
    # 0. Convert to Sparse


    if not sp.issparse(P_q1q2):
        P_q1q2 = sp.coo_matrix(P_q1q2)
    else:
        P_q1q2 = P_q1q2.tocoo()

    q1_sums = q1_values.sum(axis=1, keepdims=True)
    q1_sums[q1_sums < 1e-9] = 1.0 
    q1_values_norm = q1_values / q1_sums

    q2_sums = q2_values.sum(axis=1, keepdims=True)
    q2_sums[q2_sums < 1e-9] = 1.0 
    q2_values_norm = q2_values / q2_sums

    # 1. Preprocessing & Sparse Matrix Setup
    # ------------------------------------------------------
    rows = P_q1q2.row  # Indices corresponding to q1 (u_i)
    cols = P_q1q2.col  # Indices corresponding to q2 (v_j)
    probs = P_q1q2.data
    
    N, _ = q1_values.shape
    M, _ = q2_values_norm.shape
    num_active_pairs = len(probs) # Number of active (non-zero probability) pairs

    if debug:
        print("="*50)
        print(f"Solving Optimal Transformation with N={N}, M={M}, K={K}, Active Pairs={num_active_pairs}")
        print("="*50)

    # Calculate marginals P(q1) and P(q2)
    P_q1 = np.array(P_q1q2.sum(axis=1)).flatten()
    P_q2 = np.array(P_q1q2.sum(axis=0)).flatten()
    
    # Clip to avoid division by zero (numerical stability)
    P_q1_safe = np.clip(P_q1, 1e-9, None)
    P_q2_safe = np.clip(P_q2, 1e-9, None)

    # 2. Build Aggregation Matrices for Vectorized Operations
    # ------------------------------------------------------
    # Variable X will have shape (num_active_pairs, K), representing mu(u_i, v_j) for each active pair.
    
    # Matrix A_agg (N, num_active_pairs): Maps active pairs X to f(q1)
    # Formula: f(u_i) = sum_j P(v_j | u_i) * X_ij
    # P(v_j | u_i) = P(u_i, v_j) / P(u_i)
    A_data = probs / P_q1_safe[rows]
    A_agg = sp.coo_matrix((A_data, (rows, np.arange(num_active_pairs))), shape=(N, num_active_pairs))
    
    # Matrix B_agg (M, num_active_pairs): Maps active pairs X to calibration constraint on q2
    # Formula: sum_i P(u_i | v_j) * X_ij = v_j
    # P(u_i | v_j) = P(u_i, v_j) / P(v_j)
    B_data = probs / P_q2_safe[cols]
    B_agg = sp.coo_matrix((B_data, (cols, np.arange(num_active_pairs))), shape=(M, num_active_pairs))

    # 3. Define the Convex Optimization Problem (QP)
    # ------------------------------------------------------
    # X represents the latent ground truth distribution for each (q1, q2) pair
    X = cp.Variable((num_active_pairs, K)) 
    
    # -- Objective --
    # Calculate f(q1) based on current X
    f_q1_pred = A_agg @ X 
    
    # We want to minimize weighted MSE: sum_i P(u_i) * || f(u_i) - u_i ||^2
    # This is equivalent to minimizing Euclidean norm of: sqrt(P(u_i)) * (f(u_i) - u_i)
    sqrt_weights = np.sqrt(P_q1)
    # Reshape weights to (N, 1) for broadcasting
    diff = f_q1_pred - q1_values_norm
    weighted_diff = cp.multiply(sqrt_weights[:, None], diff)
    
    objective = cp.Minimize(cp.sum_squares(weighted_diff))
    
    # -- Constraints --
    constraints = []
    
    # Constraint 1: q2 is calibrated implies E[X | q2] == q2
    constraints.append( B_agg @ X <= q2_values_norm + tolerance )
    constraints.append( B_agg @ X >= q2_values_norm - tolerance )
    
    # Constraint 2: X must be probability distributions (Simplex)
    constraints.append( X >= 0 )
    constraints.append( cp.sum(X, axis=1) == 1 )

    if debug:
        print(f"Number of variables: {num_active_pairs * K}")
        print(f"Number of constraints: {len(constraints)}")
        print("Constraint details:")
        print(f"  1. Calibration (M x K): {M} x {K} equalities")
        print(f"  2. Non-negative: {num_active_pairs * K} inequalities")
        print(f"  3. Sum-to-1: {num_active_pairs} equalities")
    
    # 4. Solve
    # ------------------------------------------------------
    problem = cp.Problem(objective, constraints)
    
    try:
        # Use OSQP as it handles sparse QPs efficiently
        if debug:
            print("Invoking Solver (OSQP)...")
        problem.solve(solver=cp.OSQP, verbose=debug, eps_abs=2e-4, max_iter=20000)
    except Exception as e:
        print(f"Primary solver failed: {e}. Trying default...")
        problem.solve(solver=cp.OSQP, verbose=True, eps_abs=2e-4, max_iter=20000)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f">> Warning: Problem status is {problem.status}")

    if debug:
        print(f"\n--- Solver Status: {problem.status} ---")
        print(f"Optimal Value: {problem.value}")

    # 5. Debug Infeasibility Analysis
    if debug and problem.status in ["infeasible", "infeasible_inaccurate"]:
        print("\n!!! PROBLEM INFEASIBLE - ANALYZING CONSTRAINTS !!!")
        
        if X.value is None:
            print("Solver did not return a value for X. Cannot compute residuals.")
            print("Common causes:")
            print("1. P(q2) is extremely small but q2_values is not zero.")
            print("2. The problem geometry is impossible (Calibration constraint conflicts with Simplex).")
        
    elif debug and X.value is not None:
        # 即使求解是 optimal，也可以看一下约束满足的精度
        print("\n--- Constraint Check (Max Violations) ---")
        
        # Check Calibration: B_agg @ X - q2_values
        calib_residual = B_agg @ X.value - q2_values_norm
        print(f"Max Calibration Violation: {np.max(np.abs(calib_residual)):.6e}")
        
        # Check Sum-to-1: sum(X) - 1
        sum_residual = np.sum(X.value, axis=1) - 1
        print(f"Max Sum-to-1 Violation:    {np.max(np.abs(sum_residual)):.6e}")
        
        # Check Non-negative
        test_min = np.min(X.value)
        print(f"Min value in X (should be >=0): {test_min:.6e}")
        
        if np.max(np.abs(calib_residual)) > 1e-3:
            print(">> WARNING: Calibration constraint violation is high!")

    # 6. Extract Results
    # ------------------------------------------------------
    # The optimized f(q1) is the 'prediction' variable computed from optimal X
    f_optimized_values = f_q1_pred.value
    
    # Handle numerical noise / cleanup
    if f_optimized_values is None:
        return {}
        
    f_optimized_values = np.maximum(f_optimized_values, 0)
    # Normalize rows to ensure they sum to 1
    row_sums = f_optimized_values.sum(axis=1, keepdims=True)
    # Avoid division by zero for rows that might be all zero (inactive q1)
    row_sums[row_sums < 1e-9] = 1.0 
    f_optimized_values = f_optimized_values / row_sums
    
    # For q1 indices that were never visited (P(q1)=0), we default to identity map (q1)
    # (Though in result_map, we just store indices 0..N-1)
    
    result_map = {i: f_optimized_values[i] for i in range(N)}
        
    return result_map
