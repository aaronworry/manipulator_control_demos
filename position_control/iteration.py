import numpy as np
import pinocchio



def inverse_kinematics_local_frame(cmodel, cdata, id, current_q, target_dir, target_pos, dt = 1e-2):
    """
    迭代法求解关节速度，积分之后得到位置，进行机械臂位置控制
    :param cmodel: pinocchio从mjcf或urdf文件读取的机械臂模型
    :param cdata: pinocchio从mjcf或urdf文件读取的机械臂模型
    :param id: 末端执行器对应的关节，对于piper， id = 6
    :param current_q: 当前的角度值
    :param target_dir: 3x3 matrix, 末端执行器的姿态在机械臂基坐标系下的描述
    :param target_pos: 3-vector， 末端执行器的位置在机械臂基坐标系下的描述
    :param dt: 帧间的时间差，即控制频率的倒数
    :return: 机械臂关节位置控制指令
    """

    # 指定要控制的关节 ID
    JOINT_ID = id
    # 定义期望的位姿，使用目标姿态的旋转矩阵和目标位置创建 SE3 对象
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    # 将当前关节角度赋值给变量 q，作为迭代的初始值
    q = current_q
    # 定义收敛阈值，当误差小于该值时认为算法收敛
    eps = 1e-4
    # 定义最大迭代次数，防止算法陷入无限循环
    IT_MAX = 1000
    # 定义积分步长，用于更新关节角度
    DT = dt
    # 定义阻尼因子，用于避免矩阵奇异
    damp = 1e-12

    # 初始化迭代次数为 0
    i = 0
    while True:
        # 进行正运动学计算，得到当前关节角度下机器人各关节的位置和姿态
        pinocchio.forwardKinematics(cmodel, cdata, q)

        # 计算目标位姿到当前位姿之间的变换
        # oMd = oMi @ iMd
        iMd = cdata.oMi[JOINT_ID].actInv(oMdes)
        # 通过李群对数映射将变换矩阵转换为 6 维误差向量（包含位置误差和方向误差），用于量化当前位姿与目标位姿的差异
        # 目的是使 iMd = np.eye(4)
        err = pinocchio.log(iMd).vector

        # 判断误差是否小于收敛阈值，如果是则认为算法收敛
        if np.linalg.norm(err) < eps:
            success = True
            break
        # 判断迭代次数是否超过最大迭代次数，如果是则认为算法未收敛
        if i >= IT_MAX:
            success = False
            break

        # 计算当前关节角度下的雅可比矩阵，关节速度与末端速度的映射关系 (local frame)
        # J = dv_d_dq
        J = pinocchio.computeJointJacobian(cmodel, cdata, q, JOINT_ID)

        # 对雅可比矩阵进行变换，转换到李代数空间，以匹配误差向量的坐标系，同时取反以调整误差方向
        J = -np.dot(pinocchio.Jlog6(iMd), J)
        # J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)    # 和上面效果一致

        # 使用阻尼最小二乘法求解关节速度
        v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))

        # 零空间方法稳定关节值
        qpos_err = np.mod(current_q - q + np.pi, 2 * np.pi) - np.pi

        # I - J‘ J
        null_space = np.eye(8) - (J.T @ np.linalg.pinv(J.dot(J.T) + damp * np.eye(6))) @ J
        v += null_space @ qpos_err

        # 根据关节速度更新关节角度
        q = pinocchio.integrate(cmodel, q, v * DT)

        i += 1

    return q.flatten().tolist()
