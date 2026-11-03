import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def optimal_mode_mapping(P_r1_r2, r1_vals, r2_vals, K, debug=False):
    """
    Compute the optimal mapping f: r1 -> theta (mode forecast) that maximizes
    the worst-case improvement in 0-1 accuracy, given a joint distribution
    P(r1, r2) and assuming model 2 (r2) is calibrated in the sense that r2
    is a mode of P(theta | r2).

    Parameters
    ----------
    P_r1_r2 : 2D list or array, shape (n1, n2)
        Joint distribution P[r1 = r1_vals[i], r2 = r2_vals[j]].
        Must sum to 1 (up to numerical tolerance).
    r1_vals : array-like, length n1
        All possible reported values r1 can take.
        Assumed to be elements of {0, ..., K-1}.
    r2_vals : array-like, length n2
        All possible reported values r2 can take.
        Assumed to be elements of {0, ..., K-1}.
    K : int
        Size of outcome space Θ = {0, ..., K-1}.
    debug : bool, default False
        If True, prints diagnostic information.

    Returns
    -------
    obj_val : float or None
        Optimal objective value from the LP.
    f_mapping : list of int
        A list where f_mapping[i] is the optimal remapping for r1_vals[i].
        Length is len(r1_vals).
    """

    # Convert inputs to numpy arrays
    P = np.array(P_r1_r2, dtype=float)
    r1_vals = np.asarray(r1_vals, dtype=int)
    r2_vals = np.asarray(r2_vals, dtype=int)
    n1, n2 = P.shape

    if len(r1_vals) != n1 or len(r2_vals) != n2:
        raise ValueError(f"Dimensions of P_r1_r2 {P.shape} must match lengths of r1_vals ({len(r1_vals)}) and r2_vals ({len(r2_vals)}).")

    # Basic checks
    total_prob = P.sum()
    if not np.isclose(total_prob, 1.0, atol=1e-6):
        raise ValueError(f"P_r1_r2 must sum to 1. Currently sums to {total_prob}.")

    # Outcome space Θ = {0, ..., K-1}
    theta_vals = list(range(K))
    theta_index = {val: idx for idx, val in enumerate(theta_vals)}

    # Map r1_vals, r2_vals to theta indices
    try:
        r1_idx = np.array([theta_index[v] for v in r1_vals], dtype=int)
        r2_idx = np.array([theta_index[v] for v in r2_vals], dtype=int)
    except KeyError as e:
        raise ValueError(f"r1_vals and r2_vals must be in 0..K-1. Unknown value: {e.args[0]}")

    # Marginals over r1 and r2
    P_r1 = P.sum(axis=1)  # shape (n1,)
    P_r2 = P.sum(axis=0)  # shape (n2,)

    # Identify active indices where marginals are > 0 (to avoid division by 0)
    active_i = [i for i in range(n1) if P_r1[i] > 0]
    active_j = [j for j in range(n2) if P_r2[j] > 0]

    if len(active_i) == 0:
        # Degenerate: no mass at all -> trivial mapping
        if debug:
            print("All P(r1) are zero; returning identity mapping.")
        return {r1_vals[i]: r1_vals[i] for i in range(n1)}

    # Map each active i to an index for v_i
    i_to_v = {i: k for k, i in enumerate(active_i)}
    num_v = len(active_i)

    # Variables:
    # q_{theta, i, j} for theta in 0..K-1, i in 0..n1-1, j in 0..n2-1
    # plus v_i for i in active_i.
    #
    # We flatten q with index:
    #   idx_q(theta, i, j) = theta + K * (j + n2 * i).
    #
    # So q is a vector of length num_q = K * n1 * n2, then v's follow.
    num_q = K * n1 * n2
    num_vars = num_q + num_v

    def q_index(theta, i, j):
        return theta + K * (j + n2 * i)

    # Objective:
    # minimize sum_i P_r1[i] * v_i  -  sum_{i,j} q_{theta=r1_i, i, j}
    c = np.zeros(num_vars, dtype=float)

    # v_i term
    for i in active_i:
        v_idx = num_q + i_to_v[i]
        c[v_idx] = P_r1[i]

    # -q_{r1_i,i,j} term
    for i in range(n1):
        theta_r1 = r1_idx[i]
        # If P_r1[i] = 0, q_{theta,i,j} will be constrained to 0 anyway.
        for j in range(n2):
            q_idx = q_index(theta_r1, i, j)
            c[q_idx] -= 1.0

    # Equality constraints: for each (i,j),
    #   sum_theta q_{theta, i, j} = P[i,j]
    num_eq = n1 * n2
    eq_rows = []
    eq_cols = []
    eq_data = []
    b_eq = np.zeros(num_eq, dtype=float)

    row = 0
    for i in range(n1):
        for j in range(n2):
            for theta in range(K):
                eq_rows.append(row)
                eq_cols.append(q_index(theta, i, j))
                eq_data.append(1.0)
            b_eq[row] = P[i, j]
            row += 1

    A_eq = sparse.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(num_eq, num_vars))

    # Inequality constraints A_ub x <= b_ub

    ub_rows = []
    ub_cols = []
    ub_data = []
    b_ub_list = []

    # 1) Calibration constraints:
    # For each j with P_r2[j] > 0, and for all theta:
    #   sum_i q_{r2_j, i, j} >= sum_i q_{theta, i, j}
    # => -sum_i q_{r2_j, i, j} + sum_i q_{theta, i, j} <= 0
    for j in active_j:
        theta_star = r2_idx[j]
        for theta in range(K):
            row_index = len(b_ub_list)
            # sum_i q_{theta, i, j}
            for i in range(n1):
                idx_theta = q_index(theta, i, j)
                ub_rows.append(row_index)
                ub_cols.append(idx_theta)
                ub_data.append(1.0)
            # -sum_i q_{theta_star, i, j}
            for i in range(n1):
                idx_star = q_index(theta_star, i, j)
                ub_rows.append(row_index)
                ub_cols.append(idx_star)
                ub_data.append(-1.0)
            b_ub_list.append(0.0)

    # 2) v_i constraints:
    # For each i with P_r1[i] > 0, and for each theta:
    #   v_i * P_r1[i] >= sum_j q_{theta, i, j}
    # => sum_j q_{theta, i, j} - P_r1[i] * v_i <= 0
    for i in active_i:
        v_idx = num_q + i_to_v[i]
        for theta in range(K):
            row_index = len(b_ub_list)
            # sum_j q_{theta, i, j}
            for j in range(n2):
                q_idx = q_index(theta, i, j)
                ub_rows.append(row_index)
                ub_cols.append(q_idx)
                ub_data.append(1.0)
            # -P_r1[i] * v_i
            ub_rows.append(row_index)
            ub_cols.append(v_idx)
            ub_data.append(-P_r1[i])
            b_ub_list.append(0.0)

    A_ub = sparse.csr_matrix(
        (ub_data, (ub_rows, ub_cols)),
        shape=(len(b_ub_list), num_vars)
    )
    b_ub = np.array(b_ub_list, dtype=float)

    # Variable bounds: q >= 0, v_i in [0, 1]
    bounds = [(0.0, None)] * num_vars
    for i in active_i:
        v_idx = num_q + i_to_v[i]
        bounds[v_idx] = (0.0, 1.0)

    if debug:
        print("\n" + "="*60)
        print("Starting Optimal Mode Mapping Solver")
        print("="*60)
        print(f"LP setup:")
        print(f"  Grid dimensions: n1={n1}, n2={n2}, K={K}")
        print(f"  Number of q variables: {num_q}")
        print(f"  Number of v variables: {num_v}")
        print(f"  Total variables: {num_vars}")
        print(f"  Equality constraints: {num_eq}")
        print(f"  Inequality constraints: {len(b_ub_list)}")
        print(f"  Non-zeros in A_eq: {A_eq.nnz}")
        print(f"  Non-zeros in A_ub: {A_ub.nnz}")
        print(f"  Active r1 indices (P_r1 > 0): {len(active_i)}/{n1}")
        print(f"  Active r2 indices (P_r2 > 0): {len(active_j)}/{n2}")

    # Solve LP
    if debug:
        print("\nSolving LP with HiGHS...")
    
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    if not res.success:
        print(f"LP solver failed with status: {res.status}")
        print(f"Message: {res.message}")
        return None, []

    if debug:
        print(f"\nLP solved successfully!")
        print(f"  Status: {res.status}")
        print(f"  Optimal objective value: {res.fun:.6f}")
        print(f"  Iterations: {res.nit if hasattr(res, 'nit') else 'N/A'}")

    x_opt = res.x
    q_opt = x_opt[:num_q]

    # Reshape q_opt into shape (n1, n2, K) with indexing consistent with q_index
    # We used idx = theta + K * (j + n2 * i), which is equivalent to
    # reshape (n1, n2, K) in C order.
    q_array = q_opt.reshape((n1, n2, K))

    # Compute P(theta | r1 = r1_vals[i]) for each i with P_r1[i] > 0
    # alpha[theta, i] = sum_j q_{theta, i, j} / P_r1[i]
    alpha = np.zeros((K, n1), dtype=float)
    for i in active_i:
        if P_r1[i] > 0:
            for theta in range(K):
                alpha[theta, i] = q_array[i, :, theta].sum() / P_r1[i]

    # Optimal mapping f*(r1_i) = argmax_theta alpha[theta, i]
    # Return as list to match confidence.py format
    f_mapping = []

    for i in range(n1):
        if P_r1[i] > 0:
            theta_star = int(np.argmax(alpha[:, i]))
            f_mapping.append(theta_vals[theta_star])
        else:
            # r1 never occurs; mapping is irrelevant. Use identity.
            f_mapping.append(r1_vals[i])

    if debug:
        print("\nOptimal mapping f(r1):")
        for i, r1_val in enumerate(r1_vals):
            print(f"  f(r1={r1_val}) = {f_mapping[i]}")
        print("="*60 + "\n")

    return res.fun, f_mapping
