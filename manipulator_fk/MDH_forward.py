import numpy as np

class Arm_FK():
    def __init__(self, nq, dh_list = None):
        # 可动关节数量
        self.nq = nq

        # DH参数列表，每一个元素为一个3元素列表，按照alpha, a, d排列
        self.DH_list = dh_list

        # 关节零位对应的角度 theta_0
        self.theta_list = np.array([0.] * self.nq)

    def set_theta_list(self, theta_list):
        self.theta_list = theta_list

    def set_DH(self, dh_list):
        assert len(dh_list) == self.nq
        self.DH_list = dh_list

    def forward_compute(self, q):
        """
        计算末端执行器的位姿SE3，在机械臂基座坐标系下的描述
        :param q: position of all joints, a list of float
        :return: the state of the ee, including position and orientation

        Note: all representation are defined in the arm-base coordinate system
        """
        T = np.eye(4)
        for i in range(self.nq):
            Ti = self.computeTij(self.DH_list[i], q[i] + self.theta_list[i])
            T = T @ Ti
            print("=========="+str(i))
            print(Ti)
        return T

    def computeTij(self, dh, q):
        # MDH参数下的关节变换矩阵 T^i_{i-1}
        # q = 机器人的current_q + dh.theta

        alpha, a, d = dh[0], dh[1], dh[2]

        T = np.zeros((4, 4))

        calpha = np.cos(alpha)
        salpha = np.sin(alpha)
        ctheta = np.cos(q)
        stheta = np.sin(q)

        T[0, 0] = ctheta
        T[0, 1] = -stheta
        T[0, 2] = 0
        T[0, 3] = a

        T[1, 0] = stheta * calpha
        T[1, 1] = ctheta * calpha
        T[1, 2] = -salpha
        T[1, 3] = -salpha * d

        T[2, 0] = stheta * salpha
        T[2, 1] = ctheta * salpha
        T[2, 2] = calpha
        T[2, 3] = calpha * d

        T[3, 0] = 0
        T[3, 1] = 0
        T[3, 2] = 0
        T[3, 3] = 1
        # link始端的坐标系 -> 沿link平移a -> 旋转joint twist  alpha -> 沿关节平移d -> 关节转动 q -> 下一个link的坐标系
        # Tran_x(a) Rot_x(alpha) Trans_z(d) Rot_z(q)
        #=Rot_x(alpha) Tran_x(a) Rot_z(q) Trans_z(d)

        # MDH, 坐标系i建立在关节i处, DH[0].alpha = 0
        return T


if __name__ == "__main__":
    MDH_list = [[0., 0, 0.176], [-1.5708, 0, 0], [1.5708, 0, 0.264], [-1.5708, 0, 0], [1.5708, 0, 0.246], [-1.5708, 0, 0], [1.5708, 0, 0]]

    fk = Arm_FK(7, MDH_list)
    fk.set_theta_list([0.]*7)
    lj = fk.forward_compute([0.]*7)
