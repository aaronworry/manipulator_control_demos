# manipulator_control

## 项目架构
```
├─ docs:     文档
├─ example:  一些控制案例
|     ├─ function_in_mujoco.py：                  使用mujoco内置函数（计算jacobian）,实现零空间方法控制机械臂运动
|     ├─ numerical_method_with_pinocchio.py：     使用pinocchio中的函数,实现零空间方法控制机械臂运动
|     ├─ optimization_with_casadi_mdh.py：        使用casadi优化框架，根据实体机器人dh参数，实现末端执行器位姿控制
|     ├─ optimization_with_casadi_SE3.py：        使用casadi优化框架，根据仿真机器人配置文件参数，实现末端执行器位姿控制
|     ├─ optimization_with_pinocchio.py：         使用pinocchio中的优化模块，实现末端执行器位置和姿态控制
├─ manipulator_fk:   机械臂正运动学
|     ├─ MDH_forward.py：     使用MDH描述方式计算正运动学，比如piper
|     ├─ pin_fk.py：          使用pinocchio读取mjcf或urdf文件，计算正运动学
|     ├─ SDH_forward.py：     使用SDH描述方式计算正运动学
├─ manipulator_ik:   机械臂逆运动学
|     ├─ casadi_ik.py：       使用casadi优化框架求解逆运动学，基于仿真机器人配置文件参数
|     ├─ casadi_mdh_ik.py：   使用casadi优化框架求解逆运动学，基于实体机器人dh参数; (包含零空间方法案例)
|     ├─ pin_ik.py：          使用pinocchio优化模块求解逆运动学
├─ position_control:          机械臂位置控制的方法
|     ├─ iteration.py：       使用pinocchio中的函数,实现零空间方法计算关节位置
├─ position_control:          机械臂位置控制的方法
|     ├─ matrix_mjcf.py：     根据mjcf的关节参数计算关节相对位姿变换矩阵，包含piper案例
|     ├─ matrix_urdf.py：     根据urdf的关节参数计算关节相对位姿变换矩阵，包含piper案例
```
