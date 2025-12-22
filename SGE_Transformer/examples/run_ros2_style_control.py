import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from SGE_Transformer.utils.enums import DroneModel, Physics

# 尝试导入 DSLPIDControl
try:
    from SGE_Transformer.control.DSLPIDControl import DSLPIDControl
except ImportError:
    try:
        from SGE_Transformer.control.SimplePIDControl import SimplePIDControl as DSLPIDControl
    except ImportError:
        print("Error: Could not import a PID Controller.")
        exit()

from SGE_Transformer.envs.coverage_lidar_aviary import CoverageAviary


def run_ros2_style_control():
    # 1. 初始化
    env = CoverageAviary(gui=True, obstacles=True, user_debug_gui=False)
    drone_model = DroneModel.CF2X
    ctrl = DSLPIDControl(drone_model=drone_model)

    # 2. 控制参数 (模拟 ROS 2 cmd_vel 的限制)
    MAX_LIN_VEL_XY = 1.0  # m/s 最大水平速度
    MAX_LIN_VEL_Z = 0.5  # m/s 最大垂直速度
    MAX_YAW_RATE = 1.0  # rad/s 最大旋转速度 (约 57度/秒)

    # 状态积分器 (Integrator)
    # 我们不仅要记录目标位置，还要记录当前的“虚拟目标”
    obs, info = env.reset()
    target_pos = obs[:3].copy()  # 初始目标就是当前位置
    target_pos[2] = 1.0  # 起飞高度
    target_yaw = 0.0  # 初始航向

    # 可视化
    plt.ion()
    fig, ax = plt.subplots()
    grid_img = ax.imshow(env.coverage_grid, cmap='Greys', vmin=0, vmax=1)
    plt.title("ROS 2 Style Control (Body Frame Velocity)")

    print("""
    ========== ROS 2 风格控制 (速度模式) ==========
    逻辑：模拟 geometry_msgs/Twist cmd_vel
    按住键 = 发送速度指令 | 松开键 = 速度归零

    [W] 机头向前飞 (Linear X)
    [S] 机头向后飞 
    [A] 机身向左飘 (Linear Y)
    [D] 机身向右飘 
    [↑] 上升 (Linear Z)
    [↓] 下降
    [←] 向左自旋 (Angular Z)
    [→] 向右自旋
    [R] 重置
    ===============================================
    """)

    action = np.array([0, 0, 0, 0])

    try:
        while True:
            # --- A. 获取键盘输入 ---
            keys = p.getKeyboardEvents()

            # --- B. 模拟 ROS 2 Twist 消息 (机体坐标系速度) ---
            # 默认为 0 (松手即停，带有阻尼感)
            cmd_vel_body_x = 0.0
            cmd_vel_body_y = 0.0
            cmd_vel_z = 0.0
            cmd_yaw_rate = 0.0

            # 只有按住时才有速度
            if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
                cmd_vel_body_x = MAX_LIN_VEL_XY
            if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN:
                cmd_vel_body_x = -MAX_LIN_VEL_XY

            if ord('z') in keys and keys[ord('z')] & p.KEY_IS_DOWN:
                cmd_vel_body_y = MAX_LIN_VEL_XY
            if ord('c') in keys and keys[ord('c')] & p.KEY_IS_DOWN:
                cmd_vel_body_y = -MAX_LIN_VEL_XY

            if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                cmd_vel_z = MAX_LIN_VEL_Z
            if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                cmd_vel_z = -MAX_LIN_VEL_Z

            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                cmd_yaw_rate = MAX_YAW_RATE
            if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                cmd_yaw_rate = -MAX_YAW_RATE

            if ord('r') in keys and keys[ord('r')] & p.KEY_IS_DOWN:
                obs, info = env.reset()
                target_pos = obs[:3]
                target_pos[2] = 1.0
                target_yaw = 0.0
                print("Reset!")

            # --- C. 状态积分 (Integration)与坐标变换 (Coord Transform) ---
            # 这里的 dt 必须与 time.sleep 的时间一致，或者用 time.time() 算真实 dt
            dt = 1 / env.CTRL_FREQ

            # 1. 更新航向 (Yaw)
            target_yaw += cmd_yaw_rate * dt

            # 2. 将 机体速度 (Body Vel) 转换到 世界速度 (World Vel)
            # 旋转矩阵公式:
            # V_world_x = V_body_x * cos(yaw) - V_body_y * sin(yaw)
            # V_world_y = V_body_x * sin(yaw) + V_body_y * cos(yaw)
            c = np.cos(target_yaw)
            s = np.sin(target_yaw)

            vel_world_x = cmd_vel_body_x * c - cmd_vel_body_y * s
            vel_world_y = cmd_vel_body_x * s + cmd_vel_body_y * c

            # 3. 更新目标位置 (Position = Old Position + Velocity * dt)
            target_pos[0] += vel_world_x * dt
            target_pos[1] += vel_world_y * dt
            target_pos[2] += cmd_vel_z * dt

            # --- D. PID 控制 ---
            drone_state = obs[:20]

            # 构造目标速度向量 (Feed Forward)，帮助 PID 响应更快
            target_vel_world = np.array([vel_world_x, vel_world_y, cmd_vel_z])

            rpm, _, _ = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                     state=drone_state,
                                                     target_pos=target_pos,
                                                     target_rpy=np.array([0, 0, target_yaw]),
                                                     target_vel=target_vel_world,  # 传入目标速度作为前馈
                                                     target_rpy_rates=np.zeros(3))
            action = rpm

            # --- E. 环境更新 ---
            obs, reward, terminated, truncated, info = env.step(action)

            if reward > 0:
                print(f"Coverage Reward: {reward:.4f}")

            time.sleep(dt)  # 保持实时性

            if terminated:
                print("Crash! Resetting...")
                obs, info = env.reset()
                target_pos = obs[:3]
                target_pos[2] = 1.0
                target_yaw = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        plt.close()


if __name__ == "__main__":
    run_ros2_style_control()