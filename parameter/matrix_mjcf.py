import numpy as np
from scipy.spatial.transform import Rotation as R

def quat2se3(quat, pos):
    """
    handle the pos and quat of body item

    由四元数( w,x,y,z )和平移向量(x,y,z)计算固定SE3矩阵 T_body
    :param quat: 列表/数组，[w, x, y, z]（MuJoCo格式）
    :param pos: 列表/数组，[x, y, z]
    :return: T_body (4×4 numpy数组)
    """
    # 四元数转3×3旋转矩阵（scipy默认w在后，需转换为[x,y,z,w]）
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    R_mat = r.as_matrix()
    # 构造平移矩阵和旋转矩阵，复合为T_body
    T_t = np.eye(4)
    T_t[:3, 3] = pos
    T_r = np.eye(4)
    T_r[:3, :3] = R_mat
    T_body = T_t @ T_r  # 先旋转后平移，矩阵右乘
    return T_body

def axis_angle2se3(axis, theta):
    """
    handle q of joint item

    由旋转轴和角度计算关节旋转的SE3矩阵 T_joint(θ)
    :param axis: 列表/数组，[a_x, a_y, a_z]（旋转轴，已归一化）
    :param theta: 浮点数，关节旋转角度（rad）
    :return: T_joint (4×4 numpy数组)
    """
    # 轴角转3×3旋转矩阵
    r = R.from_rotvec(theta * np.array(axis))
    R_mat = r.as_matrix()
    # 构造纯旋转SE3矩阵
    T_joint = np.eye(4)
    T_joint[:3, :3] = R_mat
    return T_joint

def get_joint_se3(body_quat, body_pos, axis, theta):
    """
    主函数：输入MJCF参数和关节角度，获取相对位姿SE3矩阵
    :param body_quat: MJCF body的quat参数 [w,x,y,z]
    :param body_pos: MJCF body的pos参数 [x,y,z]
    :param axis: MJCF joint的axis参数 [a_x,a_y,a_z]
    :param theta: 关节旋转角度（rad）
    :return: T_total (4×4 numpy数组)，SE3相对位姿矩阵
    """
    # 步骤1：计算T_body
    T_body = quat2se3(body_quat, body_pos)
    # 步骤2：计算T_joint(θ)
    T_joint = axis_angle2se3(axis, theta)
    # 步骤3：复合得到最终SE3
    T_total = T_body @ T_joint
    return T_total


if __name__ == "__main__":
    # import sys
    # sys.path.append("..")
    # from manipulator_fk.pin_fk import Arm_fk

    # fk = Arm_fk("joint6")
    # fk.buildFromMJCF("../assets/snackGrasp_scene/PIPER/PIPER.xml")
    # T_ref = fk.forward(np.array([0.]*8))
    # print(T_ref)

    quat = [0.500002, -0.499999, 0.500001, 0.499997]
    pos = [1.11352e-06, 0.236, 0.246]
    axis = [0, 0, 1]
    theta = 0.
    T1 = get_joint_se3(quat, pos, axis, theta)

    quat = [0.707105, -0.707108, 0, 0]
    pos = [0, 0, 0.]
    axis = [0, 0, -1]
    theta = 0.
    T2 = get_joint_se3(quat, pos, axis, theta)

    quat = [0.707105, 0.707108, 0, 0]
    pos = [0.0, -0.264, 0]
    axis = [0, 0, 1]
    theta = 0.
    T3 = get_joint_se3(quat, pos, axis, theta)

    quat = [0.707105, -0.707108, 0, 0]
    pos = [0., 0., 0.]
    axis = [0, 0, -1]
    theta = 0.
    T4 = get_joint_se3(quat, pos, axis, theta)

    quat = [0.707105, 0.707108, 0, 0]
    pos = [0, -0.246, 0.]
    axis = [0, 0, 1]
    theta = 0.
    T5 = get_joint_se3(quat, pos, axis, theta)

    quat = [0.707105, -0.707108, 0, 0]
    pos = [0, 0, 0]
    axis = [0, 0, -1]
    theta = 0.
    T6 = get_joint_se3(quat, pos, axis, theta)

    quat = [0.707105, 0.707108, 0, 0]
    pos = [0, 0, 0]
    axis = [1, 0, 0]
    theta = 0.
    T7 = get_joint_se3(quat, pos, axis, theta)
    
    T_pra = T1@T2@T3@T4@T5@T6@T7
    # print(T_pra)
    
    # print(np.linalg.inv(T_ref) @ T_pra)
    zero_SE3_list = [T1, T2, T3, T4, T5, T6, T7]
    # axis_list = [np.array([0., 0., 1.])] * 6
    # q_min = np.array([-2.618, 0., -2.967, -1.745, -1.32, -2.094])
    # q_max = np.array([2.618, 3.14, 0., 1.745, 1.32, 2.094])

    print(T1, T2, T3, T4, T5, T6, T7)

