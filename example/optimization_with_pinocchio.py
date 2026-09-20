import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
sys.path.append("..")
from manipulator_ik.pin_ik import Arm_ik




def rotation_matrix(rx, ry, rz):
    Rx = np.array([[1., 0., 0.], [0., np.cos(rx), -np.sin(rx)], [0., np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0., np.sin(ry)], [0., 1., 0.], [-np.sin(ry), 0., np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0.], [np.sin(rz), np.cos(rz), 0.], [0., 0., 1.]])
    return Rx @ Ry @ Rz

filename = "../assets/snackGrasp_scene/PIPER/PIPER.xml"
iksolver = Arm_ik("joint6")
iksolver.buildFromMJCF(filename)

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
            self.new_q = data.qpos[:8]
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
