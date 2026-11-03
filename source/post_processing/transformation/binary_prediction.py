import numpy as np
import osqp
from scipy.sparse import csr_matrix, csc_matrix, vstack, identity

def solve_calibration_qp(P_f1f2: np.ndarray, F1_vals: np.ndarray, F2_vals: np.ndarray, debug: bool = False):
    """
    通过二次规划求解一个最优的条件概率分布 P(O=1 | F1, F2)。

    该函数寻找一个条件概率矩阵 p_ij = P(O=1 | F1=f1_i, F2=f2_j)，使其满足：
    1. F2相对于O是校准的: E[O | F2=f2_j] = f2_j for all j.
    2. F1相对于O的校准误差 E[(E[O | F1=f1_i] - f1_i)^2] 被最小化。

    Args:
        P_f1f2 (np.ndarray): 一个 I x J 的二维数组，P_f1f2[i, j] = P(F1=F1_vals[i], F2=F2_vals[j])。
        F1_vals (np.ndarray): 一个长度为 I 的一维数组，包含 F1 的所有可能取值。
        F2_vals (np.ndarray): 一个长度为 J 的一维数组，包含 F2 的所有可能取值。
        debug (bool): 如果为 True，则打印中间结果以供调试。

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None, None]:
            如果求解成功, 返回一个元组 (p_optimal, f_mapping):
            - p_optimal (np.ndarray): 一个 I x J 的二维数组, 即 P(O=1 | F1, F2)。
            - f_mapping (np.ndarray): 一个长度为 I 的一维数组, f_mapping[i] = E[O | F1=F1_vals[i]]。
            如果求解失败, 返回 (None, None)。
    """
    I, J = P_f1f2.shape
    num_vars = I * J
    P_f1f2 = np.maximum(P_f1f2, 0)

    if np.any(~np.isfinite(P_f1f2)) or np.any(P_f1f2 < 0):
        raise ValueError("P_f1f2 must be finite and non-negative.")

    total_mass = P_f1f2.sum()
    if total_mass <= 0:
        raise ValueError("P_f1f2 must have positive total mass.")
    # Normalize exactly
    P_f1f2 = P_f1f2 / total_mass

    if F1_vals.shape[0] != I or F2_vals.shape[0] != J:
        raise ValueError("Shapes mismatch: F1_vals/F2_vals must match P_f1f2 dimensions.")

    if debug:
        print("--- DEBUG MODE: solve_calibration_qp ---")
        print(f"Input shapes: P_f1f2={P_f1f2.shape}, F1_vals={F1_vals.shape}, F2_vals={F2_vals.shape}")
        print(f"Number of optimization variables (I*J): {num_vars}")
        print("-" * 40)

    # --------------------------------------------------------------------
    # 步骤 1: 构建二次规划的目标函数
    # --------------------------------------------------------------------
    P_f1 = P_f1f2.sum(axis=1)
    
    t = np.zeros(I)
    data_L = []
    row_L = []
    col_L = []

    for i in range(I):
        p_f1_i = P_f1[i]
        if p_f1_i <= 0.0:
            continue
        w_i = max(p_f1_i, 1e-6)
        inv_sqrt = 1.0 / np.sqrt(w_i)
        for j in range(J):
            val = P_f1f2[i, j] * inv_sqrt
            if abs(val) > 1e-15:
                data_L.append(val)
                row_L.append(i)
                col_L.append(i * J + j)
        t[i] = np.sqrt(w_i) * F1_vals[i]

    L = csr_matrix((data_L, (row_L, col_L)), shape=(I, num_vars))
    P_mat = 2.0 * (L.T @ L)
    q_vec = -2.0 * (L.T @ t)

    if debug:
        print("--- Step 1: Target Function Components ---")
        print(f"P_f1 (marginal prob of F1): shape={P_f1.shape}")
        print(f"L matrix: shape={L.shape}")
        print(f"t vector: shape={t.shape}")
        print(f"P_mat (QP quadratic term): shape={P_mat.shape}")
        print(f"q_vec (QP linear term): shape={q_vec.shape}")
        print("-" * 40)

    # --------------------------------------------------------------------
    # 步骤 2: 构建线性约束条件
    # --------------------------------------------------------------------
    P_f2 = P_f1f2.sum(axis=0)

    b_eq = np.zeros(J)
    data_A = []
    row_A = []
    col_A = []
    
    for j in range(J):
        p_f2_j = P_f2[j]
        if p_f2_j <= 0.0:
            continue
        w_j = max(p_f2_j, 1e-6)
        b_eq[j] = F2_vals[j]
        for i in range(I):
            val = P_f1f2[i, j] / w_j
            if abs(val) > 1e-15:
                data_A.append(val)
                row_A.append(j)
                col_A.append(i * J + j)
            
    A_eq = csc_matrix((data_A, (row_A, col_A)), shape=(J, num_vars))

    A_bounds = identity(num_vars)
    A_constr = vstack([A_eq, A_bounds], format='csc')
    
    l_bound = np.zeros(num_vars)
    u_bound = np.ones(num_vars)
    
    lower_bounds = np.concatenate([b_eq, l_bound])
    upper_bounds = np.concatenate([b_eq, u_bound])
    
    if debug:
        print("--- Step 2: Constraint Components ---")
        print(f"P_f2 (marginal prob of F2): shape={P_f2.shape}")
        print(f"A_eq (F2 calibration constraints): shape={A_eq.shape}")
        print(f"b_eq (F2 calibration targets): shape={b_eq.shape}")
        print(f"A_constr (Combined constraint matrix): shape={A_constr.shape}")
        print(f"lower_bounds: shape={lower_bounds.shape}")
        print(f"upper_bounds: shape={upper_bounds.shape}")
        print("-" * 40)

    # --------------------------------------------------------------------
    # 步骤 3: 使用OSQP求解
    # --------------------------------------------------------------------
    P_sparse = csc_matrix(P_mat)
    prob = osqp.OSQP()
    prob.setup(P=P_sparse, q=q_vec, A=A_constr, l=lower_bounds, u=upper_bounds, verbose=debug, max_iter=20000, polish=True)

    # Warm start with p_ij = f2_j (clipped to bounds)
    p0 = np.tile(np.clip(F2_vals, 0.0 + 1e-6, 1.0 - 1e-6), I)
    prob.warm_start(x=p0)

    if debug:
        print("--- Step 3: Solving with OSQP ---")
        print("OSQP solver output will be displayed below if verbose=True...\n")

    result = prob.solve()

    if debug:
        print("\nOSQP solve completed.")
        print(f"Solver status: {result.info.status}")

    if result.info.status == 'solved':
        p_optimal = result.x.reshape((I, J))
        
        # --------------------------------------------------------------------
        # 新增步骤 4: 根据 p_optimal 构建映射 f
        # f(f1_i) = E[O | F1=f1_i] = Σ_j p_optimal[i,j] * P(F2=f2_j | F1=f1_i)
        # --------------------------------------------------------------------
        
        # 为了数值稳定性，当 P_f1[i] 为 0 时，我们将其设为一个很小的数。
        # 在这种情况下，P_f1f2[i,:] 也应为 0，所以结果不会受影响。
        P_f1_safe = np.where(P_f1 > 1e-12, P_f1, 1.0)
        
        # 计算 P(F2 | F1) 矩阵
        # P_f1f2 (I,J) / P_f1_safe[:, np.newaxis] (I,1) -> 广播除法
        p_f2_given_f1 = P_f1f2 / P_f1_safe[:, np.newaxis]
        
        # 计算 f_mapping
        # (p_optimal * p_f2_given_f1) -> (I, J) 矩阵逐元素相乘
        # .sum(axis=1) -> 沿着 j 轴求和
        f_mapping = (p_optimal * p_f2_given_f1).sum(axis=1)

        if debug:
            print("\n--- Step 4: Post-processing (Build mapping f) ---")
            print(f"p_f2_given_f1 matrix: shape={p_f2_given_f1.shape}\n{p_f2_given_f1}\n")
            print(f"f_mapping vector: shape={f_mapping.shape}\n{f_mapping}\n")
            print("--- END DEBUG MODE ---\n")
            
        return p_optimal, f_mapping
    else:
        print(f"QP solver failed with status: {result.info.status}")
        if debug:
            print("--- END DEBUG MODE ---\n")
        return None, None
