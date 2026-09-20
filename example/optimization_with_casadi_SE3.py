import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
sys.path.append("..")
from manipulator_ik.casadi_ik import Arm_ik
from scipy.spatial.transform import Rotation as R



def rotation_matrix(rx, ry, rz):
    Rx = np.array([[1., 0., 0.], [0., np.cos(rx), -np.sin(rx)], [0., np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0., np.sin(ry)], [0., 1., 0.], [-np.sin(ry), 0., np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0.], [np.sin(rz), np.cos(rz), 0.], [0., 0., 1.]])
    return Rx @ Ry @ Rz

piper_q_min = np.array([-2.618, 0., -2.967, -1.745, -1.32, -2.094])
piper_q_max = np.array([2.618, 3.14, 0., 1.745, 1.32, 2.094])

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

quat_list = [[0.707105, 0, 0, -0.707108], [0.499998, 0.5, -0.500002, -0.5], [0.998726, 0, 0, 0.0504536], [0.544767, -0.544769, -0.450809, 0.450808], [0.707105, 0.707108, 0, 0], [0, 0, -0.707105, -0.707108]]
pos_list = [[0, 0, 0.123], [0, 0, 0.], [0.28358, 0.028726, 0], [-0.24221, 0.068514, 0], [0, 0, 0.], [0, 0.091, 0.0014165]]

axis_list = [np.array([0., 0., 1.])] * 6

initial_SE3_list = [quat2se3(quat_list[i], pos_list[i]) for i in range(6)]


iksolver = Arm_ik(initial_SE3_list, axis_list, piper_q_min, piper_q_max)

model = mujoco.MjModel.from_xml_path('../assets/snackGrasp_scene/PIPER/scene.xml')
data = mujoco.MjData(model)

class CustomViewer:
    def __init__(self, model, data):
        self.handle = mujoco.viewer.launch_passive(model, data)
        self.pos = 0.0001

        # 初始关节角度
        self.initial_q = data.qpos[:8].copy()

        print(f"Initial joint positions: {self.initial_q}")

        self.R_x = rotation_matrix(0., 1.5708, 0.)
        self.T = np.eye(4)
        self.T[:3, :3] = self.R_x

        self.x = 0.2
        self.new_q = self.initial_q

        self.joint_names = ["item_1/joint1", "item_1/joint2", "item_1/joint3", "item_1/joint4", "item_1/joint5", "item_1/joint6"]


    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport

    def run_loop(self):
        status = 0
        while self.is_running():
            mujoco.mj_forward(model, data)
            self.new_q = data.qpos[:6]
            print(iksolver.forward(self.new_q))
            if self.x < 0.5:
                self.x += 0.001
                self.T[:3, 3] = np.array([self.x, 0., 0.2])
                new_q = iksolver.ik(self.T, self.new_q)

            time.sleep(0.01)
            for i in range(6):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.joint_names[i])
                data.ctrl[joint_id] = new_q[i] # + self.target_angles[i]
            mujoco.mj_step(model, data)
            status += 1
            self.sync()


viewer = CustomViewer(model, data)
viewer.cam.distance = 3
viewer.cam.azimuth = 0
viewer.cam.elevation = -30
viewer.run_loop()
