import numpy as np
from ISE_Transformer.utils.enums import DroneModel

try:
    from ISE_Transformer.control.DSLPIDControl import DSLPIDControl
except ImportError:
    try:
        from ISE_Transformer.control.SimplePIDControl import SimplePIDControl as DSLPIDControl
    except ImportError:
        print("Error: Could not import a PID Controller.")
        exit()

class ROS2VelocityController:
    def __init__(self, drone_model=DroneModel.CF2X):
        self.pid = DSLPIDControl(drone_model=drone_model)

        # 内部状态积分器
        self.target_pos = None
        self.target_yaw = 0.0

        # 限制参数 (模拟真实无人机的物理限制)
        self.MAX_VEL_XY = 1.0  # 最大水平速度 (m/s)
        self.MAX_VEL_Z = 0.5  # 最大垂直速度
        self.MAX_YAW_RATE = 0.5  # 最大旋转速度 (rad/s)

    def reset(self, current_pos, current_yaw):
        """重置控制器状态，对齐当前无人机位置"""
        self.target_pos = current_pos.copy()
        self.target_yaw = current_yaw

    def compute_action(self, env_step_freq, obs, target_vel_body):
        """
        模拟 ROS 10_e 的 cmd_vel 控制逻辑
        :param env_step_freq: 控制频率 (例如 30Hz)
        :param obs: 环境观测值 (用于获取当前物理状态)
        :param target_vel_body: 目标机体速度 [vx, vy, vz, yaw_rate]
        """
        dt = 1 / env_step_freq

        # 1. 限制输入速度 (Clamping)
        vx = np.clip(target_vel_body[0], -self.MAX_VEL_XY, self.MAX_VEL_XY)
        vy = np.clip(target_vel_body[1], -self.MAX_VEL_XY, self.MAX_VEL_XY)
        vz = np.clip(target_vel_body[2], -self.MAX_VEL_Z, self.MAX_VEL_Z)
        yaw_rate = np.clip(target_vel_body[3], -self.MAX_YAW_RATE, self.MAX_YAW_RATE)

        # 10_e. 坐标转换：机体速度 -> 世界速度
        # 无人机在转，"向前"的方向也在变
        c_yaw = np.cos(self.target_yaw)
        s_yaw = np.sin(self.target_yaw)

        vel_world_x = vx * c_yaw - vy * s_yaw
        vel_world_y = vx * s_yaw + vy * c_yaw

        # 3. 积分更新目标位置 (Integration)
        # 这里的 target_pos 是平滑移动的，不会突变
        self.target_pos[0] += vel_world_x * dt
        self.target_pos[1] += vel_world_y * dt
        self.target_pos[2] += vz * dt
        self.target_yaw += yaw_rate * dt

        # 4. 安全限制 (防止飞出地图或钻地)
        # 假设地图范围大概是 -5 到 5，高度 0.10_e 到 3.0
        self.target_pos[0] = np.clip(self.target_pos[0], -4.8, 4.8)
        self.target_pos[1] = np.clip(self.target_pos[1], -4.8, 4.8)
        self.target_pos[2] = np.clip(self.target_pos[2], 0.3, 2.5)  # 最低0.3米，防止撞地坠机

        # 5. 调用底层 PID
        # 注意：我们需要从 obs 提取前20维物理状态
        drone_state = obs[:20]

        # 前馈速度 (Feed Forward)：告诉PID目标正在移动，这能极大减少拖尾和摇摆
        target_vel_world = np.array([vel_world_x, vel_world_y, vz])

        rpm, _, _ = self.pid.computeControlFromState(
            control_timestep=dt,
            state=drone_state,
            target_pos=self.target_pos,
            target_rpy=np.array([0, 0, self.target_yaw]),
            target_vel=target_vel_world,  # 关键：传入目标速度作为前馈
            target_rpy_rates=np.zeros(3)
        )

        return rpm, self.target_pos  # 返回动作和当前的虚拟目标位置