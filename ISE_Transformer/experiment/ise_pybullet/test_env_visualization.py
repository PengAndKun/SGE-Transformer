import time
import numpy as np
import pybullet as p
from ISE_Transformer.envs.coveragel_multi_room_2 import CoverageAviary  # 确保这里引入的是你刚刚修改的文件名


def test_visualization():
    print("正在初始化环境...")

    # 1. 创建环境
    # gui=True: 开启渲染窗口
    # obstacles=True: 生成我们刚刚写的复杂墙壁
    env = CoverageAviary(
        gui=True,
        obstacles=True,
        grid_res=0.5,
        radar_radius=6.0,  # 雷达半径
        num_rays=120,  # 射线数量
        rows=40, cols=40,  # 地图大小 20m x 20m
        viz_rays_every=1,  # 每步都画射线，看起来更流畅
        viz_points_every=5  # 每5步刷新一次地面绿点
    )

    # 重置环境 (生成墙壁、天花板)
    obs, info = env.reset(seed=42)

    # 将无人机放置在一个相对安全的位置 (避免出生就卡在墙里)
    # 假设地图中心是 (0,0)，我们放左下角一点
    start_pos = np.array([0, 0, 0.5])
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], start_pos, [0, 0, 0, 1])
    env.pos[0] = start_pos

    # 调整相机视角，方便观察全貌
    p.resetDebugVisualizerCamera(
        cameraDistance=12.0,
        cameraYaw=45,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0]
    )

    print("\n=== 测试开始 ===")
    print("观察重点:")
    print("1. [天花板] 是否有一层淡蓝色的半透明顶盖？")
    print("2. [红线] 无人机周围是否有红色的扫描射线？")
    print("3. [蓝点] 射线扫到墙壁时，是否留下蓝色的点？")
    print("4. [绿点] 地面是否随着移动出现绿色覆盖点？")
    print("------------------------------------------------")

    # 手动控制序列 (让无人机飞一圈，展示动态效果)
    # 格式: [dx, dy, dz]
    actions = [
        [0.1, 0, 0],  # 向前 X+
        [0, 0.1, 0],  # 向左 Y+
        [-0.1, 0, 0],  # 向后 X-
        [0, -0.1, 0],  # 向右 Y-
        [0, 0, 0.1],  # 向上 Z+ (测试天花板碰撞)
        [0, 0, -0.1],  # 向下 Z-
    ]

    # 循环运行
    try:
        for phase in range(6):  # 6个动作阶段
            move_vec = np.array(actions[phase])
            print(f"阶段 {phase + 1}: 执行动作 {move_vec} ...")

            for step in range(30):  # 每个动作持续 30 步
                current_pos = env.pos[0]
                # 计算新位置 (简单的位移控制)
                next_pos = current_pos + move_vec * 0.5  # 速度系数

                # 调用环境的 compute_scan_at_pos (包含位移+雷达+绘图)
                # 注意：这里我们直接用 compute_scan_at_pos 模拟瞬移式控制，方便测试
                # 如果你想测物理引擎 PID，可以使用 env.step()
                _, reward, terminated, _, _ = env.compute_scan_at_pos(next_pos)

                # 渲染延时 (否则太快看不清)
                time.sleep(0.05)

                if terminated:
                    print(f"!!! 发生碰撞 (Step {step}) !!!")
                    # 碰撞后稍微退回来一点，防止卡死
                    env.pos[0] = current_pos
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], current_pos, [0, 0, 0, 1])

    except KeyboardInterrupt:
        print("测试手动停止。")

    print("测试结束，关闭环境。")
    time.sleep(2)
    env.close()


if __name__ == "__main__":
    test_visualization()