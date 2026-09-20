"""
End-effector ADMM–MPC on SE(3) with joint-space first-order kinematics and MDH forward kinematics.

- Task-space error on SE(3): position in R^3 + so(3) log map for orientation
- Forward kinematics via standard MDH (same convention as MDH_forward.Arm_FK)
- Joint dynamics: only angle q and angular velocity dq; q_{k+1} = q_k + delta_k * dq_k * dt
- MPC decision / output is dq with shape (H_p, nq): one velocity per interval k = 0..H_p-1
- ADMM coupling uses augmented Lagrangian on the kinematic q residual only
"""

# 各种形式的mpc都可以构成general的形式
# 单个目标点的控制： N_ref = 1；或者N_ref > 1且所有参考点的值相等
# 上游提供参考点的数量不足：最后的参考点可以复制直至N_ref填满
# 控制周期能够使机器人到达参考点更远的未知，此时delta_var会优化至0

# 上游可以提供更稀疏的参考点，mpc部分可以根据到位检测删掉队列中的参考点

from __future__ import annotations

import casadi as ca
import numpy as np


def matrixlog3(SO3):
    """SO(3) -> so(3) as 3-vector (CasADi)."""
    acosinput = (ca.trace(SO3) - 1.0) / 2.0
    zero_vec = ca.MX.zeros(3, 1)
    omega1 = (1.0 / ca.sqrt(2 * (1 + SO3[2, 2]))) * ca.vertcat(
        SO3[0, 2], SO3[1, 2], 1 + SO3[2, 2]
    )
    omega2 = (1.0 / ca.sqrt(2 * (1 + SO3[1, 1]))) * ca.vertcat(
        SO3[0, 1], 1 + SO3[1, 1], SO3[2, 1]
    )
    omega3 = (1.0 / ca.sqrt(2 * (1 + SO3[0, 0]))) * ca.vertcat(
        1 + SO3[0, 0], SO3[1, 0], SO3[2, 0]
    )
    omega_pi = ca.if_else(
        ca.fabs(1 + SO3[2, 2]) > 1e-6,
        omega1,
        ca.if_else(ca.fabs(1 + SO3[1, 1]) > 1e-6, omega2, omega3),
    )
    pi_vec = ca.pi * omega_pi
    theta = ca.arccos(acosinput)
    so3_mat = theta / (2.0 * ca.sin(theta)) * (SO3 - SO3.T)
    normal_vec = ca.vertcat(so3_mat[2, 1], so3_mat[0, 2], so3_mat[1, 0])
    return ca.if_else(
        acosinput >= 1.0, zero_vec, ca.if_else(acosinput <= -1.0, pi_vec, normal_vec)
    )


def mdh_Tij(dh, q_ca):
    """Single MDH link transform T_i^{i-1}, same as MDH_forward.Arm_FK.buildTij."""
    alpha, a, d = float(dh[0]), float(dh[1]), float(dh[2])
    return ca.blockcat(
        [
            [ca.cos(q_ca), -ca.sin(q_ca), 0, a],
            [
                ca.sin(q_ca) * ca.cos(alpha),
                ca.cos(q_ca) * ca.cos(alpha),
                -ca.sin(alpha),
                -d * ca.sin(alpha),
            ],
            [
                ca.sin(q_ca) * ca.sin(alpha),
                ca.cos(q_ca) * ca.sin(alpha),
                ca.cos(alpha),
                d * ca.cos(alpha),
            ],
            [0, 0, 0, 1],
        ]
    )


def fk_se3_mdh(q_ca, dh_list, theta_offset=None):
    """
    End-effector pose T_{ee} in base frame, q_ca: (nq,) CasADi symbol, dh_list length nq.
    """
    nq = len(dh_list)
    if theta_offset is None:
        theta_offset = [0.0] * nq
    T = ca.DM.eye(4)
    for i in range(nq):
        Ti = mdh_Tij(dh_list[i], q_ca[i] + theta_offset[i])
        T = ca.mtimes(T, Ti)
    return T


def se3_pose_error(T_cur, T_des, w_pos=1.0, w_rot=1.0):
    """
    6D error: [e_p; e_R], e_p = p_des - p_cur (world), e_R = log(R_cur^T R_des).
    Returns scalar weighted squared norm for cost and vector for debugging.
    """
    R_c = T_cur[:3, :3]
    R_d = T_des[:3, :3]
    p_c = T_cur[:3, 3]
    p_d = T_des[:3, 3]
    e_p = p_d - p_c
    e_R = matrixlog3(ca.mtimes(R_c.T, R_d))
    e = ca.vertcat(e_p, e_R)
    cost = w_pos * ca.sumsqr(e_p) + w_rot * ca.sumsqr(e_R)
    return cost, e


class EEADMM_MPC:
    """
    ADMM splitting for multi-reference SE(3) tracking with first-order joint kinematics (q, dq only).
    Optimizes dq in R^{H_p x nq} (one joint velocity per stage); q_{k+1} = q_k + delta_k * dq_k * dt.
    """

    def __init__(
        self,
        dt,
        nq,
        dh_list,
        N_ref,
        H_p,
        pose_lambda,
        time_lambda,
        smooth_lambda,
        w_pos,
        w_rot,
        rho,
        epsilon_pri,
        epsilon_dual,
        max_iter,
        theta_offset=None,
        rho_max=100.0,
        rho_inc=1.2,
    ):
        self.dt = float(dt)
        self.nq = int(nq)
        self.dh_list = list(dh_list)
        assert len(self.dh_list) == self.nq
        self.theta_offset = theta_offset if theta_offset is not None else [0.0] * self.nq
        self.N_ref = int(N_ref)
        self.H_p = int(H_p)
        self.pose_lambda = float(pose_lambda)
        self.time_lambda = float(time_lambda)
        self.smooth_lambda = float(smooth_lambda)
        self.w_pos = float(w_pos)
        self.w_rot = float(w_rot)
        self.rho = float(rho)
        self.rho_max = float(rho_max)
        self.rho_inc = float(rho_inc)
        self.epsilon_pri = float(epsilon_pri)
        self.epsilon_dual = float(epsilon_dual)
        self.max_iter = int(max_iter)

        self.q_min = self.q_max = None
        self.dq_min = self.dq_max = None
        self.M_big = 1.0

        self.residual_pri = []
        self.residual_dual = []

        self._built = False

    def set_bounds(self, q_bounds, dq_bounds):
        self.q_min = np.asarray(q_bounds[0], dtype=float).reshape(self.nq)
        self.q_max = np.asarray(q_bounds[1], dtype=float).reshape(self.nq)
        self.dq_min = np.asarray(dq_bounds[0], dtype=float).reshape(self.nq)
        self.dq_max = np.asarray(dq_bounds[1], dtype=float).reshape(self.nq)
        self.M_big = 10.0 * max(float(np.max(np.abs(self.dq_max))) * self.dt, 1e-6)

    def _fk(self, q_row):
        qv = ca.vertcat(*[q_row[i] for i in range(self.nq)])
        return fk_se3_mdh(qv, self.dh_list, self.theta_offset)

    def _pose_cost(self, q_row, T_ref_col):
        T_tar = ca.reshape(T_ref_col, 4, 4)
        T_cur = self._fk(q_row)
        c, _ = se3_pose_error(T_cur, T_tar, self.w_pos, self.w_rot)
        return c

    def ca_q_next(self, q, dq, delta):
        """Discrete joint angle update from angular velocity dq (first-order only)."""
        return q + delta * dq * self.dt

    def build_solver(self):
        if self.q_min is None:
            raise RuntimeError("Call set_bounds() before build_solver().")
        q_lo = ca.DM(self.q_min).reshape((1, self.nq))
        q_hi = ca.DM(self.q_max).reshape((1, self.nq))
        dq_lo = ca.DM(self.dq_min).reshape((1, self.nq))
        dq_hi = ca.DM(self.dq_max).reshape((1, self.nq))
        ones_1xnq = ca.DM.ones(1, self.nq)

        opt_sc = ca.Opti()
        q = opt_sc.variable(self.H_p + 1, self.nq)
        dq = opt_sc.variable(self.H_p, self.nq)

        delta_param = opt_sc.parameter(self.H_p)
        y_param = opt_sc.parameter(self.H_p, self.nq)
        alpha_param = opt_sc.parameter(self.H_p + 1, self.N_ref)
        T_ref_param = opt_sc.parameter(16, self.N_ref)
        q0_p = opt_sc.parameter(1, self.nq)

        opt_sc.subject_to(q[0, :] == q0_p)

        for k in range(self.H_p + 1):
            opt_sc.subject_to(opt_sc.bounded(q_lo, q[k, :], q_hi))
        for k in range(self.H_p):
            opt_sc.subject_to(opt_sc.bounded(dq_lo, dq[k, :], dq_hi))

        for k in range(self.H_p):
            q_next = self.ca_q_next(q[k, :], dq[k, :], delta_param[k])
            opt_sc.subject_to(q[k + 1, :] == q_next)

            q_diff = q[k + 1, :] - q[k, :]
            opt_sc.subject_to(q_diff <= self.M_big * delta_param[k] * ones_1xnq)
            opt_sc.subject_to(-q_diff <= self.M_big * delta_param[k] * ones_1xnq)
        for k in range(self.H_p - 1):
            dq_diff = dq[k + 1, :] - dq[k, :]
            opt_sc.subject_to(dq_diff <= self.M_big * delta_param[k] * ones_1xnq)
            opt_sc.subject_to(-dq_diff <= self.M_big * delta_param[k] * ones_1xnq)

        obj_sc = 0.0
        for k in range(self.H_p + 1):
            for j in range(self.N_ref):
                Tcol = T_ref_param[:, j]
                obj_sc += (
                    self.pose_lambda
                    * alpha_param[k, j]
                    * self._pose_cost(q[k, :], Tcol)
                )
        for k in range(self.H_p - 1):
            obj_sc += self.smooth_lambda * ca.sumsqr(dq[k + 1, :] - dq[k, :])
        for k in range(self.H_p):
            q_next = self.ca_q_next(q[k, :], dq[k, :], delta_param[k])
            res = q[k + 1, :] - q_next
            obj_sc += (
                ca.mtimes(y_param[k, :], res.T)
                + 0.5 * self.rho * ca.sumsqr(res)
            )

        opt_sc.minimize(obj_sc)
        opt_sc.solver(
            "ipopt",
            {
                "ipopt.max_iter": 1000,
                "ipopt.tol": 1e-6,
                "ipopt.print_level": 0,
                "print_time": 0,
            },
        )

        opt_ay = ca.Opti()
        alpha_var = opt_ay.variable(self.H_p + 1, self.N_ref)
        delta_var = opt_ay.variable(self.H_p)
        q_param = opt_ay.parameter(self.H_p + 1, self.nq)
        dq_param = opt_ay.parameter(self.H_p, self.nq)
        y_param2 = opt_ay.parameter(self.H_p, self.nq)
        T_ref_param2 = opt_ay.parameter(16, self.N_ref)

        for j in range(self.N_ref):
            opt_ay.subject_to(ca.sum1(alpha_var[:, j]) == 1.0)
            for k in range(self.H_p + 1):
                opt_ay.subject_to(alpha_var[k, j] >= 0.0)

        for k in range(self.H_p):
            opt_ay.subject_to(delta_var[k] >= 0.0)
            opt_ay.subject_to(delta_var[k] <= 1.0)
            q_diff = q_param[k + 1, :] - q_param[k, :]
            mb = self.M_big * delta_var[k]
            opt_ay.subject_to(q_diff <= mb * ones_1xnq)
            opt_ay.subject_to(-q_diff <= mb * ones_1xnq)
        for k in range(self.H_p - 1):
            dq_diff = dq_param[k + 1, :] - dq_param[k, :]
            mb = self.M_big * delta_var[k]
            opt_ay.subject_to(dq_diff <= mb * ones_1xnq)
            opt_ay.subject_to(-dq_diff <= mb * ones_1xnq)

        for k in range(self.H_p - 1):
            opt_ay.subject_to(delta_var[k + 1] <= delta_var[k])
        opt_ay.subject_to(delta_var[self.H_p - 1] == 0.0)

        obj_ay = self.time_lambda * ca.sum1(delta_var) * self.dt
        obj_ay += 2.0 * ca.sum1(delta_var * (1.0 - delta_var))
        for k in range(self.H_p + 1):
            for j in range(self.N_ref):
                Tcol = T_ref_param2[:, j]
                obj_ay += (
                    self.pose_lambda
                    * alpha_var[k, j]
                    * self._pose_cost(q_param[k, :], Tcol)
                )
        for k in range(self.H_p):
            q_next = self.ca_q_next(
                q_param[k, :], dq_param[k, :], delta_var[k]
            )
            res = q_param[k + 1, :] - q_next
            obj_ay += (
                ca.mtimes(y_param2[k, :], res.T)
                + 0.5 * self.rho * ca.sumsqr(res)
            )

        opt_ay.minimize(obj_ay)
        opt_ay.solver(
            "ipopt",
            {
                "ipopt.max_iter": 500,
                "ipopt.tol": 1e-6,
                "ipopt.print_level": 0,
                "print_time": 0,
            },
        )

        self._opt_sc = opt_sc
        self._opt_ay = opt_ay

        self._var_q, self._var_dq = q, dq
        self._p_delta_sc, self._p_y_sc, self._p_alpha_sc = delta_param, y_param, alpha_param
        self._p_Tref_sc = T_ref_param
        self._p_q0 = q0_p

        self._var_alpha, self._var_delta = alpha_var, delta_var
        self._p_q_ay, self._p_dq_ay = q_param, dq_param
        self._p_y_ay = y_param2
        self._p_Tref_ay = T_ref_param2
        self._built = True

    def _Tref_to_param(self, T_list):
        P = np.zeros((16, self.N_ref))
        for j, T in enumerate(T_list):
            P[:, j] = np.asarray(T, dtype=float).reshape(16, order="F")
        return P

    def solve(
        self,
        q0,
        T_ref_list,
        q_init,
        dq_init,
        alpha_init,
        delta_init,
        y_init,
        verbose=True,
    ):
        """
        Solve ADMM. Primary control output is joint velocity dq with shape (H_p, nq).
        Control dq is free (only box bounds); warm-start via dq_init / set_initial.
        """
        if not self._built:
            self.build_solver()
        T_flat = self._Tref_to_param(T_ref_list)
        q = np.asarray(q_init, dtype=float).reshape(self.H_p + 1, self.nq)
        dq = np.asarray(dq_init, dtype=float).reshape(self.H_p, self.nq)
        alpha = np.asarray(alpha_init, dtype=float)
        delta = np.asarray(delta_init, dtype=float).reshape(self.H_p)
        refine_delta = (delta > 0.5).astype(float)
        y = np.asarray(y_init, dtype=float)

        self.residual_pri = []
        self.residual_dual = []

        rho = self.rho
        for it in range(self.max_iter):
            if verbose:
                print(f"\n===== ADMM iteration {it + 1}/{self.max_iter} =====")

            self._opt_sc.set_value(self._p_Tref_sc, T_flat)
            self._opt_sc.set_value(
                self._p_q0, np.asarray(q0, dtype=float).reshape(1, self.nq)
            )
            self._opt_sc.set_value(self._p_y_sc, y)
            self._opt_sc.set_value(self._p_delta_sc, refine_delta)
            self._opt_sc.set_value(self._p_alpha_sc, alpha)

            self._opt_sc.set_initial(self._var_q, q)
            self._opt_sc.set_initial(self._var_dq, dq)

            try:
                sol = self._opt_sc.solve()
                q = sol.value(self._var_q)
                dq = sol.value(self._var_dq)
            except Exception as e:
                if verbose:
                    print(f"State/control subproblem failed: {e}")
                break

            self._opt_ay.set_value(self._p_Tref_ay, T_flat)
            self._opt_ay.set_value(self._p_q_ay, q)
            self._opt_ay.set_value(self._p_dq_ay, dq)
            self._opt_ay.set_value(self._p_y_ay, y)

            self._opt_ay.set_initial(self._var_alpha, alpha)
            self._opt_ay.set_initial(self._var_delta, delta)

            try:
                sol2 = self._opt_ay.solve()
                alpha = sol2.value(self._var_alpha)
                delta = sol2.value(self._var_delta)
                refine_delta = (delta > 0.5).astype(float)
            except Exception as e:
                if verbose:
                    print(f"Alpha/delta subproblem failed: {e}")
                break

            res_prim = 0.0
            y_new = y.copy()
            for k in range(self.H_p):
                q_next = self.q_next_num(q[k], dq[k], refine_delta[k])
                res = q[k + 1] - q_next
                res_prim = max(res_prim, float(np.linalg.norm(res)))
                y_new[k] = y[k] + rho * res
            res_dual = float(rho * np.linalg.norm(y_new - y))
            y = y_new

            self.residual_pri.append(res_prim)
            self.residual_dual.append(res_dual)
            rho = min(rho * self.rho_inc, self.rho_max)

            if verbose:
                print(f"Primal residual: {res_prim:.6e} | Dual residual: {res_dual:.6e}")

            if res_prim < self.epsilon_pri and res_dual < self.epsilon_dual:
                if verbose:
                    print("ADMM converged.")
                break

        return q, dq, alpha, delta, y

    def q_next_num(self, q, dq, delta):
        return q + delta * dq * self.dt

    def fk_numpy(self, q):
        """Numeric FK for inspection (4x4)."""
        T = np.eye(4)
        q = np.asarray(q, dtype=float).reshape(self.nq)
        for i in range(self.nq):
            Ti = self._mdh_Tij_numpy(self.dh_list[i], q[i] + self.theta_offset[i])
            T = T @ Ti
        return T

    @staticmethod
    def _mdh_Tij_numpy(dh, qi):
        alpha, a, d = float(dh[0]), float(dh[1]), float(dh[2])
        c, s = np.cos(qi), np.sin(qi)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array(
            [
                [c, -s, 0, a],
                [s * ca, c * ca, -sa, -d * sa],
                [s * sa, c * sa, ca, d * ca],
                [0, 0, 0, 1],
            ]
        )


if __name__ == "__main__":
    np.random.seed(0)
    nq = 6
    MDH_list = [[0., 0., 0.123], [-np.pi/2, 0., 0.], [0., 0.285, 0.], [np.pi/2, -0.022, 0.25], [-np.pi/2, 0., 0.], [np.pi/2, 0., 0.091]]
    dt = 0.02
    N_ref = 2
    H_p = 10
    mpc = EEADMM_MPC(
        dt=dt,
        nq=nq,
        dh_list=MDH_list,
        N_ref=N_ref,
        H_p=H_p,
        pose_lambda=5.0,
        time_lambda=0.5,
        smooth_lambda=0.5,
        w_pos=1.0,
        w_rot=0.5,
        rho=2.0,
        epsilon_pri=1e-3,
        epsilon_dual=1e-3,
        max_iter=40,
        theta_offset=[0.0, -np.pi * 174.22 / 180, -100.78 / 180 * np.pi, 0.0, 0.0, 0.0],
    )
    mpc.set_bounds(
        ([-2.618, 0., -2.967, -1.745, -1.32, -2.094], [2.618, 3.14, 0., 1.745, 1.32, 2.094]),
        ([-5.0] * nq, [5.0] * nq),
    )
    mpc.build_solver()

    q0 = np.zeros(nq)
    T_ref = []
    Ttemp = np.array([[0., 0., 1., 0.3], [0., 1., 0., 0.], [-1., 0., 0., 0.2], [0., 0., 0., 1.]])
    for j in range(N_ref):
        T_ref.append(Ttemp)

    q_init = np.tile(q0, (H_p + 1, 1))
    dq_init = np.zeros((H_p, nq))
    delta = np.ones(H_p)
    delta[-1] = 0.0
    alpha = np.ones((H_p + 1, N_ref)) / (H_p + 1)
    y = np.zeros((H_p, nq))

    q, dq, alpha, delta, y = mpc.solve(
        q0, T_ref, q_init, dq_init, alpha, delta, y, verbose=True
    )
    print("Final EE position:", mpc.fk_numpy(q[-1])[:3, 3])
    print("First-step joint velocity dq[0]:", dq[0])
