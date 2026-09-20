[简介](https://blog.csdn.net/m0_38122847/article/details/143781613)

## URDF主要标签
一种xml文件格式，描述机器人的构型，任意刚体机器人都可以分解为 关节（描述零件之间的相对运动） 和 连杆（零件） 的排列组合。
urdf中各个标签可以是并列的， **joint**, **link**没有嵌套关系
**link** 用于描述刚性零件，比如机械臂的连杆臂。
包含如下属性：外观属性，碰撞模型，惯性参数
&emsp;&emsp;**visual** 描述外观属性，即仿真器中的模样。可以使用标准的标签，也可以导入stl文件	
&emsp;&emsp;&emsp;&emsp;**geometry**  几何形状
&emsp;&emsp;&emsp;&emsp;**material** 颜色纹理
&emsp;&emsp;**collision** 描述碰撞模型，一般用于gazebo中的碰撞反馈
&emsp;&emsp;&emsp;&emsp;**geometry**
&emsp;&emsp;**inertial** 描述动力学模型的惯性参数
**mass** 计算平动的质量
&emsp;&emsp;**inertia** 计算旋转的转动惯量
**origin** 质心
**joint** 用于描述两个**link**之间的配合，比如旋转，平移
&emsp;&emsp;**parent** 父节点**link**, 从base作为根节点计算
&emsp;&emsp;**child** 子节点**link**, 从base计算		
&emsp;&emsp;**origin** 相对于父节点的位置和朝向
&emsp;&emsp;**axis** 旋转关节的转轴

urdf可以接入ros生态中的各种插件**plugin**，实现传感器，驱动模型的接入

[参考链接1](https://zhuanlan.zhihu.com/p/662107096)
[参考链接2](https://wiki.ros.org/urdf#Examples)

## mjcf的主要标签
mjcf描述的内容更加丰富，除了机器人模型之外，也能够集成场景的所有信息
不同于urdf在link的visual中引用资产模型，mjcf会定义一个**asset**标签记录所有需要使用的资产模型（stl， obj）
mjcf使用**body**和**joint**描述机器人构型，由于**joint**没有父子节点的概念，**body**和**joint**有严格的顺序，并使用标签嵌套的方式描述机器人构型。描述机器人构型的参数和urdf一致。
mjcf可以直接在文件中定义传感器，控制器。
[参考链接1](https://mujoco.readthedocs.io/)

### 一个mjcf可以完整地描述仿真场景，而urdf只能描述机器人构型本身。

使用urdf描述场景时，需要依赖其他文件。即ros采用一套文件系统定义仿真场景。但这种方式对多机系统更加友好；也方便文件的复用。

