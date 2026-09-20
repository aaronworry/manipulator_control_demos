import numpy as np
import casadi as ca
import os

def _nullspace_orientation_cost(
    error_vec,
    q_var,
    nq: int,
    w_rot_null: float = 0.25,
    damp: float = 1e-4,
):
    """任务优先级式代价：主任务为位置误差；姿态仅通过位置雅可比零空间内的分量进入代价。

    令 e_p、e_o 分别为位置、姿态（李代数）误差；J_p、J_o 为对 q 的雅可比。
    阻尼右伪逆 J_p^# = J_p^T (J_p J_p^T + λ I)^{-1}，零空间矩阵 N = I - J_p^# J_p。
    将关节空间“姿态修正方向” g_o = J_o^T e_o 投影为 g_null = N g_o，再惩罚 ||g_null||^2，
    使姿态优化主要在不影响一阶位置任务的子空间内进行；w_rot_null 应明显小于位置项权重。
    """
    e_p = error_vec[:3]
    e_o = error_vec[3:6]
    J_p = ca.jacobian(e_p, q_var)
    J_o = ca.jacobian(e_o, q_var)
    I_n = ca.DM.eye(nq)
    JpT = ca.transpose(J_p)
    JoT = ca.transpose(J_o)
    JpJpT = ca.mtimes(J_p, JpT) + damp * ca.DM.eye(3)
    J_p_inv_r = ca.mtimes(JpT, ca.solve(JpJpT, ca.DM.eye(3)))
    N_mat = I_n - ca.mtimes(J_p_inv_r, J_p)
    g_o = ca.mtimes(JoT, e_o)
    g_null = ca.mtimes(N_mat, g_o)
    return ca.sumsqr(e_p) + w_rot_null * ca.sumsqr(g_null)

class Arm_ik():
    def __init__(self, dh_params, q_min, q_max):
        self.dh_params = dh_params
        self.q_min = q_min
        self.q_max = q_max
        self.nq = len(self.dh_params)

        q_value = ca.SX.sym("q_value", self.nq)
        self.forward = ca.Function("forward", [q_value], [self.mdh_forward(q_value)])

    def mdh_forward(self, q):
        T = ca.DM.eye(4)
        i = 0
        for alpha, a, d, theta_zero in self.dh_params:
            T_dh = ca.blockcat([
                [ca.cos(q[i] + theta_zero), -ca.sin(q[i] + theta_zero), 0, a],
                [ca.sin(q[i] + theta_zero) * ca.cos(alpha), ca.cos(q[i] + theta_zero) * ca.cos(alpha), -ca.sin(alpha), -d * ca.sin(alpha)],
                [ca.sin(q[i] + theta_zero) * ca.sin(alpha), ca.cos(q[i] + theta_zero) * ca.sin(alpha), ca.cos(alpha), d * ca.cos(alpha)],
                [0, 0, 0, 1]
            ])
            T = ca.mtimes(T, T_dh)
            i += 1
        return T


    def matrixlog3(self, SO3):
        """
        CasADi符号版本 旋转矩阵转李代数 (SO3 -> so3)
        :param SO3: CasADi的SX/MX/DM对象，3x3的旋转矩阵（必须是合法的正交矩阵、行列式=1）
        :return: CasADi的3x1向量，so3李代数，对应旋转轴+旋转角的紧凑表示
        """
        # 替换np.trace → CasADi矩阵迹运算
        acosinput = (ca.trace(SO3) - 1.0) / 2.0

        # ========== 分支1: acosinput >= 1 ，θ=0，无旋转，返回零向量 ==========
        zero_vec = ca.SX.zeros(3, 1)

        # ========== 分支2: acosinput <= -1 ，θ=π(180°) 奇异情况 ==========
        omega1 = (1.0 / ca.sqrt(2 * (1 + SO3[2,2]))) * ca.vertcat(SO3[0,2], SO3[1,2], 1 + SO3[2,2])
        omega2 = (1.0 / ca.sqrt(2 * (1 + SO3[1,1]))) * ca.vertcat(SO3[0,1], 1 + SO3[1,1], SO3[2,1])
        omega3 = (1.0 / ca.sqrt(2 * (1 + SO3[0,0]))) * ca.vertcat(1 + SO3[0,0], SO3[1,0], SO3[2,0])

        omega_pi = ca.if_else(ca.fabs(1 + SO3[2,2]) > 1e-6, omega1,
                    ca.if_else(ca.fabs(1 + SO3[1,1]) > 1e-6, omega2, omega3))
        pi_vec = ca.pi * omega_pi

        # ========== 分支3: -1 < acosinput < 1 ，常规非奇异情况 ==========
        theta = ca.arccos(acosinput)
        so3_mat = theta / (2.0 * ca.sin(theta)) * (SO3 - SO3.T)
        # 提取反对称矩阵的李代数向量：[so3[2,1], so3[0,2], so3[1,0]] 索引和原代码一致
        normal_vec = ca.vertcat(so3_mat[2,1], so3_mat[0,2], so3_mat[1,0])

        res = ca.if_else(acosinput >= 1.0, zero_vec, ca.if_else(acosinput <= -1.0, pi_vec, normal_vec))

        return res

    def pose_error(self, q, T_tar):
        T_cur = self.mdh_forward(q)
        p_c = T_cur[:3, 3]
        p_d = T_tar[:3, 3]
        R_c = T_cur[:3, :3]
        R_d = T_tar[:3, :3]

        # 位置误差：期望减实际
        e_pos = p_d - p_c

        # 姿态误差：R_d * R_c^T 的李代数（旋转矩阵→角速度向量，误差最小）
        R_ref = R_c.T @ R_d


        e_att = self.matrixlog3(R_ref)
        return ca.vertcat(e_pos, e_att)

    def createSolver(self):
        # Creating symbolic variables
        self.cq = ca.SX.sym("q", self.nq, 1)
        self.cTf = ca.SX.sym("tf", 4, 4)


        self.error = ca.Function(
            "error",
            [self.cq, self.cTf],
            [
                self.pose_error(self.cq, self.cTf)
            ],
        )


        # Defining the optimization problem
        self.opti = ca.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.var_q_last = self.opti.parameter(self.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.regularization_cost = ca.sumsqr(self.var_q)
        self.smooth_cost = ca.sumsqr(self.var_q - self.var_q_last)

        # 误差的符号化计算
        error_vector = self.error(self.var_q, self.param_tf)
        # 零空间方法，位置误差优先
        self.error_cost = _nullspace_orientation_cost(
            error_vector, self.var_q, self.nq, w_rot_null=1., damp=1e-4
        )
        
        # pos_error = error_vector[:3]  # 位置误差
        # ori_error = error_vector[3:]  # 姿态误差
        # 设置位置和姿态的权重
        # weight_position = 1.0  # 位置权重
        # weight_orientation = 0.5  # 姿态权重
        # 误差的cost
        # self.error_cost = weight_position * ca.sumsqr(pos_error) + weight_orientation * ca.sumsqr(ori_error)

        

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.q_min,
            self.var_q,
            self.q_max)
        )

        self.opti.minimize(200.0 * self.error_cost + 0.01 * self.regularization_cost + 1. * self.smooth_cost)

        ##### IPOPT #####
        opts = {
            'ipopt':{
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4,
                'sb': 'yes'
                # 'hessian_approximation':"limited-memory"
            },
            'print_time':False  # print or not
            #'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.nq)

    def ik(self, T , current_arm_motor_q = None, current_arm_motor_dq = None):
        if current_arm_motor_q is not None:
            self.init_data = current_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf, T)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)

            if current_arm_motor_dq is not None:
                v = current_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q


            dof = np.zeros(self.nq)
            dof[:len(sol_q)] = sol_q
            return dof

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)


            if current_arm_motor_dq is not None:
                v = current_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q


            dof = np.zeros(self.nq)
            dof[:len(sol_q)] = self.init_data

            raise e


if __name__ == "__main__":
    q_min = np.array([-2.618, 0., -2.967, -1.745, -1.32, -2.094])
    q_max = np.array([2.618, 3.14, 0., 1.745, 1.32, 2.094])
    mdh_params = [[0., 0., 0.123, 0.], [-np.pi/2, 0., 0., -np.pi * 174.22 / 180], [0., 0.285, 0., -100.78 / 180 * np.pi], [np.pi/2, -0.022, 0.25, 0.], [-np.pi/2, 0., 0., 0.], [np.pi/2, 0., 0.091, 0.]]
    arm = Arm_ik(mdh_params, q_min, q_max)
    arm.createSolver()
    theta = np.pi
    tf = np.array([
            [1, 0, 0, 0.2],
            [0, np.cos(theta), -np.sin(theta), 0.0],
            [0, np.sin(theta), np.cos(theta), 0.3],
            [0, 0, 0, 1]
        ])
    dof = arm.ik(tf)
    print(f"DoF: {dof}")
    print(arm.forward(dof))

    T_ref = np.eye(4)
    T_ref[2, 3] = 0.14
    dof_new = np.array([1.1004650592803955,2.292524814605713,-0.9363167881965637,1.1231368780136108,-1.2222191095352173,-0.2865830659866333])
    print(arm.forward(dof_new)@T_ref)
