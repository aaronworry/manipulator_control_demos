import pinocchio as pin
import os
import numpy as np

class Arm_fk():
    def __init__(self, ee_frame) -> None:
        # 末端执行器对应的关节id
        self.frame_name = ee_frame

    def buildFromMJCF(self, mcjf_file):
        self.arm = pin.RobotWrapper.BuildFromMJCF(mcjf_file)

    def buildFromURDF(self, urdf_file):
        self.arm = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[os.path.dirname(urdf_file)])

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



