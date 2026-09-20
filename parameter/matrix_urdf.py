import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix(rpy):
    Rx = np.array([[1., 0., 0.], [0., np.cos(rpy[0]), -np.sin(rpy[0])], [0., np.sin(rpy[0]), np.cos(rpy[0])]])
    Ry = np.array([[np.cos(rpy[1]), 0., np.sin(rpy[1])], [0., 1., 0.], [-np.sin(rpy[1]), 0., np.cos(rpy[1])]])
    Rz = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]), 0.], [np.sin(rpy[2]), np.cos(rpy[2]), 0.], [0., 0., 1.]])
    return Rz @ Ry @ Rx


def rpy2se3(rpy, pos):
    """
    SciPy实现URDF标准RPY转3×3旋转矩阵（Z-Y-X旋转顺序，静态旋转）
    :param rpy: list/np.array，形状(3,)，URDF的rpy参数 [roll, pitch, yaw]
    :param degrees: bool，输入是否为角度制，URDF默认弧度制，故默认False
    :return: np.array，形状(3,3)，对应的旋转矩阵
    """
    # 基于Z-Y-X静态旋转创建SciPy旋转对象，匹配URDF RPY规则
    r = R.from_euler(seq='xyz', angles=rpy, degrees=False)
    # 转换为3×3旋转矩阵
    R_mat = r.as_matrix()
    # 构造平移矩阵和旋转矩阵，复合为T_body
    T_t = np.eye(4)
    T_t[:3, 3] = pos
    T_r = np.eye(4)
    T_r[:3, :3] = R_mat# rotation_matrix(rpy)
    T_body = T_t @ T_r # 先旋转后平移，矩阵右乘
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

def get_joint_se3(rpy, pos, axis, theta):
    """
    主函数：输入URDF参数和关节角度，获取相对位姿SE3矩阵
    :param rpy: URDF joint的rpy参数 [roll, pitch, yaw]
    :param pos: URDF joint的pos参数 [x,y,z]
    :param axis: URDF joint的axis参数 [a_x,a_y,a_z]
    :param theta: 关节旋转角度（rad）
    :return: T_total (4×4 numpy数组)，SE3相对位姿矩阵
    """
    # 步骤1：计算T_body
    T_body = rpy2se3(rpy, pos)
    # 步骤2：计算T_joint(θ)
    T_joint = axis_angle2se3(axis, theta)
    # 步骤3：复合得到最终SE3
    T_total = T_body @ T_joint
    return T_total


if __name__ == "__main__":

    
    rpy = [0, 0, 0]
    pos = [0, 0, 0.176]
    axis = [0, 0, 1]
    theta = 0.
    T1 = get_joint_se3(rpy, pos, axis, theta)
    
    rpy = [-1.5708, 0, 0]
    pos = [0, 0, 0]
    axis = [0, 0, -1]
    theta = 0.
    T2 = get_joint_se3(rpy, pos, axis, theta)

    rpy = [1.5708, 0, 0]
    pos = [0, -0.264, 0]
    axis = [0, 0, 1]
    theta = 0.
    T3 = get_joint_se3(rpy, pos, axis, theta)
    
    rpy = [-1.5708, 0, 0]
    pos = [0, 0, 0]
    axis = [0, 0, -1]
    theta = 0.
    T4 = get_joint_se3(rpy, pos, axis, theta)

    rpy = [1.5708, 0, 0]
    pos = [0, -0.246, 0.]
    axis = [0, 0, 1]
    theta = 0.
    T5 = get_joint_se3(rpy, pos, axis, theta)
    
    rpy = [-1.5708, 0, 0]
    pos = [0, 0, 0]
    axis = [0, 0, 1]
    theta = 0.
    T6 = get_joint_se3(rpy, pos, axis, theta)

    rpy = [1.5708, 0, 0]
    pos = [0, 0, 0]
    axis = [-1, 0, 0]
    theta = 0.
    T7 = get_joint_se3(rpy, pos, axis, theta)



    
    
    
    print(T1)
    print(T2)
    print(T3)
    print(T4)
    print(T5)
    print(T6)
    print(T7)

