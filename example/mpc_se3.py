import time
import sys
import numpy as np
import mujoco
import mujoco.viewer

sys.path.append("..")
from position_control.baseManipulatorMPC import BaseArmMPC


def rotation_matrix(rx, ry, rz):
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(rx), -np.sin(rx)], [0.0, np.sin(rx), np.cos(rx)]]
    )
    Ry = np.array(
        [[np.cos(ry), 0.0, np.sin(ry)], [0.0, 1.0, 0.0], [-np.sin(ry), 0.0, np.cos(ry)]]
    )
    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0.0], [np.sin(rz), np.cos(rz), 0.0], [0.0, 0.0, 1.0]]
    )
    return Rx @ Ry @ Rz


class CustomViewer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.handle = mujoco.viewer.launch_passive(model, data)

        self.dt = 0.02
        self.mpc_horizon = 10

        self.q_min = np.array([-2.618, 0.0, -2.967, -1.745, -1.32, -2.094])
        self.q_max = np.array([2.618, 3.14, 0.0, 1.745, 1.32, 2.094])
        self.dq_min = np.array([-5, -5, -5, -5, -5, -5])
        self.dq_max = np.array([5, 5, 5, 5, 5, 5])

        self.W_pose = np.diag([20, 20, 20, 80, 80, 80])
        self.W_u = 0.001 * np.eye(6)
        self.W_du = 0.001 * np.eye(6)

        self.mdh_params = [
            [0.0, 0.0, 0.123, 0.0],
            [-np.pi / 2, 0.0, 0.0, -np.pi * 174.22 / 180],
            [0.0, 0.285, 0.0, -100.78 / 180 * np.pi],
            [np.pi / 2, -0.022, 0.25, 0.0],
            [-np.pi / 2, 0.0, 0.0, 0.0],
            [np.pi / 2, 0.0, 0.091, 0.0],
        ]

        self.mpc = BaseArmMPC(
            self.dt, self.mpc_horizon, self.W_pose, self.W_u, self.W_du, self.mdh_params
        )
        self.mpc.construct_opti_ee(self.q_min, self.q_max, self.dq_min, self.dq_max)

        self.joint_names = [
            "item_1/joint1",
            "item_1/joint2",
            "item_1/joint3",
            "item_1/joint4",
            "item_1/joint5",
            "item_1/joint6",
        ]
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.joint_names
        ]

        self.q_cmd = self.data.qpos[:6].copy()
        self.x = 0.4
        self.target_y = 0.0
        self.target_z = 0.2

        self.T_des = np.eye(4)
        self.T_des[:3, :3] = rotation_matrix(0.0, 1.5708, 0.0)
        self.T_des[:3, 3] = np.array([self.x, self.target_y, self.target_z])

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    def run_loop(self):
        status = 0
        while self.is_running():
            mujoco.mj_forward(self.model, self.data)
            q_current = self.data.qpos[:6].copy()

            # if self.x < 0.5:
            #     self.x += 0.001
            # self.T_des[:3, 3] = np.array([self.x, self.target_y, self.target_z])

            dq_opt = self.mpc.solve_ee(q_current, self.T_des)
            self.q_cmd = np.clip(q_current + dq_opt * self.dt, self.q_min, self.q_max)
            # time.sleep(0.01)
            for i in range(6):
                self.data.ctrl[self.joint_ids[i]] = self.q_cmd[i]

            mujoco.mj_step(self.model, self.data)
            status += 1
            if status % 100 == 0:
                ee_pos = self.mpc.forward(q_current).full()[:3, 3].reshape(-1)
                pos_err = np.linalg.norm(self.T_des[:3, 3] - ee_pos)
                print(f"step={status}, target_x={self.x:.3f}, pos_err={pos_err:.4f}")

            self.sync()
            # time.sleep(0.01)


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("../assets/snackGrasp_scene/PIPER/scene.xml")
    data = mujoco.MjData(model)

    viewer = CustomViewer(model, data)
    viewer.cam.distance = 3
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -30
    viewer.run_loop()
