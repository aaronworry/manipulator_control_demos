import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import os

class Arm_ik():
    def __init__(self, ee_frame) -> None:
        self.frame_name = ee_frame

    def buildFromMJCF(self, mcjf_file):
        self.arm = pin.RobotWrapper.BuildFromMJCF(mcjf_file)
        self.createSolver()

    def buildFromURDF(self, urdf_file):
        self.arm = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[os.path.dirname(urdf_file)])
        self.createSolver()

    def createSolver(self):
        self.model = self.arm.model
        self.data = self.arm.data

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.ee_id = self.model.getFrameId(self.frame_name)

        self.error = casadi.Function(
            "error",
            # ^b_t T = ^b_6 T @ ^6_t T
            # ^6_t T = ^b_6 T.inv() @ ^b_t T
            # 目标是 ^6_t T = np.eye(4), 求矩阵对数之后即为 差距
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.ee_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.var_q_last = self.opti.parameter(self.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # 误差的符号化计算
        error_vector = self.error(self.var_q, self.param_tf)
        pos_error = error_vector[:3]  # 位置误差
        ori_error = error_vector[3:]  # 姿态误差
        # 设置位置和姿态的权重
        weight_position = 1.0  # 位置权重
        weight_orientation = 0.1  # 姿态权重

        # 误差的cost
        self.error_cost = weight_position * casadi.sumsqr(pos_error) + weight_orientation * casadi.sumsqr(ori_error)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )

        self.opti.minimize(200.0 * self.error_cost + 0.01 * self.regularization_cost + 1. * self.smooth_cost)

        ##### IPOPT #####
        opts = {
            'ipopt':{
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4,
                'sb': 'yes'
            },
            'print_time':False  # print or not
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.model.nq)

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

            dof = np.zeros(self.model.nq)
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

            dof = np.zeros(self.model.nq)
            dof[:len(sol_q)] = self.init_data

            raise e

    def forward(self, q):
        # 根据关节值，计算末端执行器的位姿，translation为位置，rotation为姿态（3x3矩阵）
        # T = [R, t; 0, 1]
        pin.forwardKinematics(self.arm.model, self.arm.data, q)
        pin.updateFramePlacements(self.arm.model, self.arm.data)
        self.ee_id = self.arm.model.getFrameId(self.frame_name)
        pose = self.arm.data.oMf[self.ee_id]
        translation = pose.translation
        rotation = pose.rotation
        result = np.eye(4)
        result[:3,:3] = rotation
        result[:3,3] = translation
        return result


if __name__ == "__main__":
    arm = Arm_ik("joint6")
    arm.buildFromURDF("../../assets/piper_description/piper_description.urdf")
    theta = np.pi
    tf = np.array([
            [1, 0, 0, 0.2],
            [0, np.cos(theta), -np.sin(theta), 0.0],
            [0, np.sin(theta), np.cos(theta), 0.3],
            [0, 0, 0, 1]
        ])
    dof = arm.ik(tf)
    print(f"DoF: {dof}")

