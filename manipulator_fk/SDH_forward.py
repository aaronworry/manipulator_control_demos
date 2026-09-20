import casadi as ca
import numpy as np

# here is version using casadi to describe the fk model

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
        return T

    def computeTij(self, dh, q):
        # SDH参数下的关节变换矩阵 T^i_{i-1}
        # q = 机器人的current_q + dh.theta

        alpha, a, d = dh[0], dh[1], dh[2]
        T_a = np.eye(4)
        T_a[0, 3] = a

        T_d = np.eye(4)
        T_d[2, 3] = d

        Rzt = np.array([[np.cos(q), -np.sin(q), 0., 0.],
            [np.sin(q), np.cos(q), 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])

        Rxa = np.array([[1., 0., 0., 0.],
            [0., np.cos(alpha), -np.sin(alpha), 0.],
            [0., np.sin(alpha), np.cos(alpha), 0.],
            [0., 0., 0., 1.]])
        """
        np.array([
                [np.cos(q), -np.sin(q)*np.cos(alpha),  np.sin(q)*np.sin(alpha), a*np.cos(q)],
                [np.sin(q),  np.cos(q)*np.cos(alpha), -np.cos(q)*np.sin(alpha), a*np.sin(q)],
                [0,              np.sin(alpha),                np.cos(alpha),               d],
                [0,              0,                            0,                           1]
                ])
        SDH 坐标系i建立在关节i+1处, DH[0].alpha = pi/2 or -pi/2
        """
        return Rzt @ T_d @ Rxa @ T_a


if __name__ == "__main__":
    SDH_list = [[0., 1., 0.], [0., 1., 0.], [0., 1., 1.]]
    fk = Arm_FK(3, SDH_list)
    T = np.eye(4)
    lj = fk.forward_compute([np.pi/6, np.pi/6, np.pi/6])
    print(lj @ T)

