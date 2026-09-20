import numpy as np
import casadi as ca




class BaseArmMPC():
    def __init__(self, dt, N, W_pose, W_u, W_du, dh_params):
        self.dt = dt            # 0.005 s  对应piper控制频率
        self.N = N

        self.nq = len(dh_params)

        self.W_pose = W_pose
        self.W_u = W_u
        self.W_du = W_du

        self.nv = self.nq

        self.dh_params = dh_params

        # 符号变量定义 (核心！MPC的优化变量和状态变量)
        self.q = ca.SX.sym('q', self.nq)        # 当前关节角度 (状态量)
        self.dq = ca.SX.sym('dq', self.nv)      # 关节速度 (控制量)
        self.T_ref = ca.SX.sym('T_ref', 4, 4)

        # 声明MPC优化变量：预测时域内的所有关节速度，形状 (N*n_joint, 1)
        self.U = ca.SX.sym('U', self.N * self.nv)

        self.forward = ca.Function('T', [self.q], [self.mdh_forward(self.q)])


    def set_T_des(self, T):
        self.T_des = T

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

    def sdh_forward(self, q):
        T = ca.DM.eye(4)
        i = 0
        for alpha, a, d, theta_zero in self.dh_params:
            # DH齐次变换矩阵公式
            T_dh = ca.blockcat([
                [ca.cos(q[i] + theta_zero), -ca.sin(q[i] + theta_zero)*ca.cos(alpha), ca.sin(q[i] + theta_zero)*ca.sin(alpha), a*ca.cos(q[i] + theta_zero)],
                [ca.sin(q[i] + theta_zero), ca.cos(q[i] + theta_zero)*ca.cos(alpha), -ca.cos(q[i] + theta_zero)*ca.sin(alpha), a*ca.sin(q[i] + theta_zero)],
                [0, ca.sin(alpha), ca.cos(alpha), d],
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
        # 嵌套if_else 实现原代码的 elif 判断逻辑
        omega_pi = ca.if_else(ca.fabs(1 + SO3[2,2]) > 1e-6, omega1,
                    ca.if_else(ca.fabs(1 + SO3[1,1]) > 1e-6, omega2, omega3))
        pi_vec = ca.pi * omega_pi

        # ========== 分支3: -1 < acosinput < 1 ，常规非奇异情况 ==========
        theta = ca.arccos(acosinput)
        so3_mat = theta / (2.0 * ca.sin(theta)) * (SO3 - SO3.T)
        # 提取反对称矩阵的李代数向量：[so3[2,1], so3[0,2], so3[1,0]] 索引和原代码一致
        normal_vec = ca.vertcat(so3_mat[2,1], so3_mat[0,2], so3_mat[1,0])

        # ========== CasADi 符号条件分支整合（核心替换python原生if/elif/else） ==========
        res = ca.if_else(acosinput >= 1.0, zero_vec, ca.if_else(acosinput <= -1.0, pi_vec, normal_vec))

        return res

    def pose_error(self, T_current, T_des):
        """
        计算末端实际位姿与期望位姿的6维误差向量 e = [e_pos, e_att]^T
        e_pos: 3维位置误差 [x_err, y_err, z_err] = p_des - p_current
        e_att: 3维姿态误差（基于旋转矩阵的李代数误差，无奇异、连续可微）
        :param T_current: 4×4 实际位姿矩阵
        :param T_des: 4×4 期望位姿矩阵
        :return: 6×1 位姿误差向量
        """
        p_c = T_current[:3, 3]
        p_d = T_des[:3, 3]
        R_c = T_current[:3, :3]
        R_d = T_des[:3, :3]

        # 位置误差：期望减实际
        e_pos = p_d - p_c

        # 姿态误差：R_d * R_c^T 的李代数（旋转矩阵→角速度向量，误差最小）
        R_ref = R_c.T @ R_d


        e_att = self.matrixlog3(R_ref)
        return ca.vertcat(e_pos, e_att)

    def construct_opti_ee(self, q_min, q_max, dq_min, dq_max):


        # 初始化MPC代价函数和约束
        cost = 0.0
        self.g = []  # 约束函数列表
        self.g_lb = []  # 约束下限
        self.g_ub = []  # 约束上限
        q_current = self.q  # 当前关节角度，作为迭代起点

        for k in range(self.N):
            # 提取第k步的关节速度控制量
            dq_k = self.U[k * self.nv: (k+1) * self.nv]

            # ① 关节角度更新：前向欧拉积分 q_{k+1} = q_k + dq_k * dt
            q_next = q_current + dq_k * self.dt

            # ② 计算当前末端位姿和位姿误差
            T_current = self.mdh_forward(q_current)
            e = self.pose_error(T_current, self.T_ref)

            # ③ 累加代价函数：位姿跟踪误差 + 控制量平滑性约束
            cost += ca.mtimes([e.T, self.W_pose, e])          # 核心：位姿跟踪代价
            cost += ca.mtimes([dq_k.T, self.W_u, dq_k])       # 关节速度惩罚，避免超速

            if k > 0:
                # 速度增量惩罚，保证控制量连续，抑制关节抖动
                dq_prev = self.U[(k-1)*self.nv : k*self.nv]
                cost += ca.mtimes([(dq_k - dq_prev).T, self.W_du, (dq_k - dq_prev)])

            # ④ 添加约束：关节角度约束 q ∈ [q_min, q_max]
            self.g.append(q_next)
            self.g_lb.extend(q_min)
            self.g_ub.extend(q_max)

            # ⑤ 添加约束：关节速度约束 dq ∈ [dq_min, dq_max]
            self.g.append(dq_k)
            self.g_lb.extend(dq_min)
            self.g_ub.extend(dq_max)

            # 更新关节角度，进入下一个预测步
            q_current = q_next

        # 构建优化问题：最小化cost，变量U，约束 g ∈ [g_lb, g_ub]
        nlp = {
            'x': self.U,          # 优化变量：N步关节速度
            'f': cost,       # 代价函数
            'g': ca.vertcat(*self.g),  # 约束向量
            'p': ca.vertcat(self.q, ca.vec(self.T_ref))           # 优化问题参数：当前关节角度（每次滚动更新）, 目标姿态
        }

        # 选择求解器：IPOPT（工业级非线性规划求解器，CasADi原生支持，无梯度消失）
        # 设置求解器参数，平衡速度与精度
        opts = {
            'ipopt.max_iter': 200,
            'ipopt.tol': 1e-6,
            'ipopt.print_level': 0,  # 0=关闭日志，1=详细日志
            'print_time': False      # 关闭求解时间打印
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)



    def solve_ee(self, q_current, T_des):
        # 优化变量初始值：全零速度，加快收敛
        U0 = np.zeros(self.N * self.nv)

        # 求解非线性规划问题
        sol = self.solver(
            x0=U0,
            lbx=-np.inf, ubx=np.inf,
            lbg=self.g_lb, ubg=self.g_ub,
            p=np.concatenate([q_current, T_des.flatten(order = 'F')])
        )

        # 提取最优解：仅执行第一步关节速度（MPC核心：滚动时域，只取第一步）
        U_opt = sol['x'].full().flatten()
        dq_opt = U_opt[:self.nv]
        return dq_opt

if __name__ == "__main__":
    q_min = np.array([-2.618, 0., -2.967, -1.745, -1.32, -2.094])
    q_max = np.array([2.618, 3.14, 0., 1.745, 1.32, 2.094])
    dq_min = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    dq_max = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    W_pose = np.diag([10, 10, 10, 100, 100, 100])  # 位姿权重(位置x3+姿态x3)，姿态权重可略低
    W_u = 0.1 * np.eye(6)              # 关节速度权重，抑制抖动
    W_du = 0.05 * np.eye(6)            # 关节速度增量权重，保证控制平滑

    mdh_params = [[0., 0., 0.123, 0.], [-np.pi/2, 0., 0., -np.pi * 174.22 / 180], [0., 0.285, 0., -100.78 / 180 * np.pi], [np.pi/2, -0.022, 0.25, 0.], [-np.pi/2, 0., 0., 0.], [np.pi/2, 0., 0.091, 0.]]

    T_des = np.array([
                [0, 0, 1, 0.3],
                [0, 1, 0, 0.0],
                [-1, 0, 0, 0.2],
                [0, 0, 0, 1]
            ])



    mpc = BaseArmMPC(0.05, 10, W_pose, W_u, W_du, mdh_params)
    mpc.construct_opti_ee(q_min, q_max, dq_min, dq_max)

    q_np = np.zeros(6)
    sim_time = 10.0  # 仿真总时间10s
    n_steps = int(sim_time / 0.05)
    print("开始机械臂末端位姿MPC跟踪控制...")
    for step in range(n_steps):
        # 1. 调用MPC控制器，得到最优关节速度
        dq_np = mpc.solve_ee(q_np, T_des)

        # 2. 执行控制：更新关节角度（实际机械臂中替换为关节驱动）
        q_np = q_np + dq_np * 0.05

        # 3. 计算当前末端实际位姿，打印跟踪误差
        T_current_np = mpc.forward(q_np).full()
        e_pos = T_des[:3,3] - T_current_np[:3,3]
        pos_err_norm = np.linalg.norm(e_pos)

        if step % 20 == 0:  # 每1s打印一次
            print(f"仿真时间: {step*0.05:.2f}s | 位置误差范数: {pos_err_norm:.4f} m")
            # print(T_current_np[:3, :3].T@T_des[:3, :3])
            print(T_current_np[:3, :3])
            if pos_err_norm < 1e-3:
                print("位姿跟踪收敛！误差小于1mm")
                break

    print("仿真结束！")
