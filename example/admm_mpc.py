import sys
import numpy as np
import mujoco
import mujoco.viewer

sys.path.append("..")
from position_control.ee_admm_mpc import EEADMM_MPC


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
        self.H_p = 10
        self.N_ref = 2
        self.nq = 6
        nq = self.nq

        self.q_min = np.array([-2.618, 0.0, -2.967, -1.745, -1.32, -2.094])
        self.q_max = np.array([2.618, 3.14, 0.0, 1.745, 1.32, 2.094])
        self.dq_min = np.array([-2.0] * nq)
        self.dq_max = np.array([2.0] * nq)

        dh_list = [
            [0.0, 0.0, 0.123],
            [-np.pi / 2, 0.0, 0.0],
            [0.0, 0.285, 0.0],
            [np.pi / 2, -0.022, 0.25],
            [-np.pi / 2, 0.0, 0.0],
            [np.pi / 2, 0.0, 0.091],
        ]
        theta_offset = [
            0.0,
            -np.pi * 174.22 / 180,
            -100.78 / 180 * np.pi,
            0.0,
            0.0,
            0.0,
        ]

        self.mpc = EEADMM_MPC(
            dt=self.dt,
            nq=nq,
            dh_list=dh_list,
            N_ref=self.N_ref,
            H_p=self.H_p,
            pose_lambda=5.0,
            time_lambda=0.5,
            smooth_lambda=0.5,
            w_pos=1.0,
            w_rot=0.5,
            rho=2.0,
            epsilon_pri=1e-2,
            epsilon_dual=1e-2,
            max_iter=12,
            theta_offset=theta_offset,
        )
        self.mpc.set_bounds(
            (self.q_min.tolist(), self.q_max.tolist()),
            (self.dq_min.tolist(), self.dq_max.tolist()),
        )
        self.mpc.build_solver()

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
        self.x = 0.3
        self.target_y = 0.0
        self.target_z = 0.2

        self.T_des = np.eye(4)
        self.T_des[:3, :3] = rotation_matrix(0.0, 1.5708, 0.0)
        self.T_des[:3, 3] = np.array([self.x, self.target_y, self.target_z])

        self._last_q = None
        self._last_dq = None
        self._alpha = np.ones((self.H_p + 1, self.N_ref)) / (self.H_p + 1)
        self._delta = np.ones(self.H_p)
        self._delta[-1] = 0.0
        self._y = np.zeros((self.H_p, self.nq))

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    def _T_ref_list(self):
        T_a = self.T_des.copy()
        T_b = self.T_des.copy()
        return [T_a, T_b]

    def _warm_q_init(self, q_meas):
        q_init = np.zeros((self.H_p + 1, self.nq))
        q_init[0] = q_meas
        if self._last_q is None:
            q_init[1:] = q_meas
            return q_init
        for k in range(1, self.H_p):
            q_init[k] = self._last_q[k + 1]
        q_init[self.H_p] = self._last_q[self.H_p]
        return q_init

    def _warm_dq_init(self):
        if self._last_dq is None:
            return np.zeros((self.H_p, self.nq))
        dq_init = np.zeros((self.H_p, self.nq))
        for k in range(self.H_p - 1):
            dq_init[k] = self._last_dq[k + 1]
        dq_init[self.H_p - 1] = self._last_dq[self.H_p - 1]
        return dq_init

    def run_loop(self):
        status = 0
        while self.is_running():
            mujoco.mj_forward(self.model, self.data)
            q_current = self.data.qpos[:6].copy()

            # if self.x < 0.5:
            #     self.x += 0.001
            # self.T_des[:3, 3] = np.array([self.x, self.target_y, self.target_z])

            q_init = self._warm_q_init(q_current)
            dq_init = self._warm_dq_init()
            T_ref = self._T_ref_list()

            try:
                q, dq, self._alpha, self._delta, self._y = self.mpc.solve(
                    q_current,
                    T_ref,
                    q_init,
                    dq_init,
                    self._alpha,
                    self._delta,
                    self._y,
                    verbose=False,
                )
                self._last_q = q
                self._last_dq = dq
                dq0 = dq[0]
            except Exception:
                dq0 = (
                    self._last_dq[0]
                    if self._last_dq is not None
                    else np.zeros(self.nq, dtype=float)
                )

            self.q_cmd = np.clip(
                q_current + dq0 * self.dt, self.q_min, self.q_max
            )
            for i in range(6):
                self.data.ctrl[self.joint_ids[i]] = self.q_cmd[i]

            mujoco.mj_step(self.model, self.data)
            status += 1
            if status % 100 == 0:
                ee_pos = self.mpc.fk_numpy(q_current)[:3, 3]
                pos_err = np.linalg.norm(self.T_des[:3, 3] - ee_pos)
                print(f"step={status}, target_x={self.x:.3f}, pos_err={pos_err:.4f}")

            self.sync()


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("../assets/snackGrasp_scene/PIPER/scene.xml")
    data = mujoco.MjData(model)

    viewer = CustomViewer(model, data)
    viewer.cam.distance = 3
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -30
    viewer.run_loop()
