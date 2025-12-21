import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType



class CoverageAviary(BaseAviary):
    def __init__(self,
                 grid_bounds=((-5, 5), (-5, 5)),
                 grid_res=0.2,
                 fov_angle=np.deg2rad(45),
                 drone_model=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq=240,
                 ctrl_freq=240,
                 gui=False,
                 record=False,
                 observationType=ObservationType.KIN,
                 user_debug_gui=True,
                 **kwargs):

        # 1. 先初始化自定义参数
        self.grid_bounds = grid_bounds
        self.grid_res = grid_res
        self.fov_angle = fov_angle

        self.rows = int((grid_bounds[0][1] - grid_bounds[0][0]) / grid_res)
        self.cols = int((grid_bounds[1][1] - grid_bounds[1][0]) / grid_res)
        self.map_size = self.rows * self.cols
        self.coverage_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # 2. 再调用父类初始化
        # 父类会调用 _actionSpace 和 _observationSpace，所以这两个方法必须在下面定义好
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=observationType,
                         user_debug_gui=user_debug_gui)

    ################################################################################
    # 必须实现的方法 1: 动作空间
    ################################################################################
    def _actionSpace(self):
        """定义动作空间：4个电机的RPM"""
        # 4个电机，范围 [0, MAX_RPM]
        act_lower_bound = np.array([0., 0., 0., 0.])
        act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################
    # 必须实现的方法 2: 观察空间
    ################################################################################
    def _observationSpace(self):
        """定义观察空间：物理状态(20维) + 覆盖地图"""
        # 物理状态 20维 (pos, quat, rpy, vel, ang_vel, last_clipped_action)
        phys_dim = 20
        return spaces.Box(
            low=0, high=1,
            shape=(phys_dim + self.map_size,),
            dtype=np.float32
        )

    ################################################################################
    # 必须实现的方法 3: 计算观测值
    ################################################################################
    def _computeObs(self):
        """返回当前的观测值"""
        # 获取第0号无人机的物理状态 [20,]
        drone_state = self._getDroneStateVector(0)

        # 展平地图
        flat_map = self.coverage_grid.flatten().astype(np.float32)

        # 拼接
        return np.concatenate([drone_state, flat_map])

    ################################################################################
    # 必须实现的方法 4: 动作预处理
    ################################################################################
    def _preprocessAction(self, action):
        """将输入动作转换为电机RPM"""
        # 如果输入已经是 RPM (我们这里假设是)，直接裁剪并返回
        # 注意：BaseAviary 期望返回 (NUM_DRONES, 4)
        return np.clip(action, 0, self.MAX_RPM)

    ################################################################################
    # 必须实现的方法 5: 计算奖励
    ################################################################################
    def _computeReward(self):
        """计算子模奖励"""
        pos = self.pos[0]
        z_height = pos[2]
        if z_height <= 0.1: return 0.0

        radius = z_height * np.tan(self.fov_angle)
        new_covered_count = self._update_grid(pos[0], pos[1], radius)
        return new_covered_count * (self.grid_res ** 2)

    ################################################################################
    # 必须实现的方法 6, 7, 8: 终止条件与Info
    ################################################################################
    def _computeTerminated(self):
        """是否结束 (例如撞地)"""
        if self.pos[0][2] < 0.05:  # 坠机
            return True
        return False

    def _computeTruncated(self):
        """是否超时 (由外部 wrapper 控制，这里返回 False)"""
        return False

    def _computeInfo(self):
        """返回调试信息"""
        return {"total_coverage": np.sum(self.coverage_grid) / (self.rows * self.cols)}

    ################################################################################
    # 覆盖父类的默认障碍物方法，创建自定义房间
    ################################################################################
    def _addObstacles(self):
        """创建自定义的房间环境：四周是墙，中间有随机柱子"""
        # 获取 PyBullet 客户端 ID
        client = self.CLIENT

        # 1. 根据 grid_bounds 计算墙的位置
        x_min, x_max = self.grid_bounds[0]
        y_min, y_max = self.grid_bounds[1]

        # 房间高度
        wall_height = 3.0
        wall_thickness = 0.5

        # 定义墙的颜色 (R, G, B, A)
        wall_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0, 0, 0], rgbaColor=[0.7, 0.7, 0.7, 1])

        # --- 创建四面墙 ---

        # 墙 1 (X_min 侧)
        # 碰撞体积：Box 的参数是半长 (halfExtents)
        col_x_min = p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=[wall_thickness, (y_max - y_min) / 2, wall_height / 2])
        vis_x_min = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=[wall_thickness, (y_max - y_min) / 2, wall_height / 2],
                                        rgbaColor=[0.8, 0.8, 0.8, 1])
        p.createMultiBody(baseMass=0,  # 0表示静止物体
                          baseCollisionShapeIndex=col_x_min,
                          baseVisualShapeIndex=vis_x_min,
                          basePosition=[x_min - wall_thickness, 0, wall_height / 2],
                          physicsClientId=client)

        # 墙 2 (X_max 侧)
        col_x_max = p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=[wall_thickness, (y_max - y_min) / 2, wall_height / 2])
        vis_x_max = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=[wall_thickness, (y_max - y_min) / 2, wall_height / 2],
                                        rgbaColor=[0.8, 0.8, 0.8, 1])
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=col_x_max,
                          baseVisualShapeIndex=vis_x_max,
                          basePosition=[x_max + wall_thickness, 0, wall_height / 2],
                          physicsClientId=client)

        # 墙 3 (Y_min 侧)
        col_y_min = p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=[(x_max - x_min) / 2 + wall_thickness * 2, wall_thickness,
                                                        wall_height / 2])
        vis_y_min = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=[(x_max - x_min) / 2 + wall_thickness * 2, wall_thickness,
                                                     wall_height / 2], rgbaColor=[0.8, 0.8, 0.8, 1])
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=col_y_min,
                          baseVisualShapeIndex=vis_y_min,
                          basePosition=[0, y_min - wall_thickness, wall_height / 2],
                          physicsClientId=client)

        # 墙 4 (Y_max 侧)
        col_y_max = p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=[(x_max - x_min) / 2 + wall_thickness * 2, wall_thickness,
                                                        wall_height / 2])
        vis_y_max = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=[(x_max - x_min) / 2 + wall_thickness * 2, wall_thickness,
                                                     wall_height / 2], rgbaColor=[0.8, 0.8, 0.8, 1])
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=col_y_max,
                          baseVisualShapeIndex=vis_y_max,
                          basePosition=[0, y_max + wall_thickness, wall_height / 2],
                          physicsClientId=client)

        # 2. 添加一些随机的障碍物 (圆柱体或立方体)
        # 这些障碍物会造成遮挡，增加探索难度，非常适合验证你的算法
        num_obstacles = 5
        np.random.seed(42)  # 固定种子保证每次环境一致

        for _ in range(num_obstacles):
            # 随机位置 (避开中心起飞点)
            obs_x = np.random.uniform(x_min + 1, x_max - 1)
            obs_y = np.random.uniform(y_min + 1, y_max - 1)
            if abs(obs_x) < 1.0 and abs(obs_y) < 1.0: continue  # 跳过原点

            # 创建一个 1x1x2 米的障碍物
            half_size = [0.5, 0.5, 1.0]
            col_obs = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)
            vis_obs = p.createVisualShape(p.GEOM_BOX, halfExtents=half_size, rgbaColor=[0.4, 0.4, 0.4, 1])

            p.createMultiBody(baseMass=0,
                              baseCollisionShapeIndex=col_obs,
                              baseVisualShapeIndex=vis_obs,
                              basePosition=[obs_x, obs_y, 1.0],  # 放在地面上
                              physicsClientId=client)

    ################################################################################

    ################################################################################
    # 辅助方法
    ################################################################################
    def _update_grid(self, x, y, radius):
        c_x = int((x - self.grid_bounds[0][0]) / self.grid_res)
        c_y = int((y - self.grid_bounds[1][0]) / self.grid_res)
        r_cells = int(radius / self.grid_res)

        x_min = max(0, c_x - r_cells)
        x_max = min(self.rows, c_x + r_cells + 1)
        y_min = max(0, c_y - r_cells)
        y_max = min(self.cols, c_y + r_cells + 1)

        if x_min >= x_max or y_min >= y_max: return 0

        Y, X = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (X - c_x) ** 2 + (Y - c_y) ** 2
        mask = dist_sq <= r_cells ** 2

        new_coverage = (self.coverage_grid[y_min:y_max, x_min:x_max] == 0) & mask
        count = np.sum(new_coverage)

        self.coverage_grid[y_min:y_max, x_min:x_max][mask] = 1
        return count

    def reset(self, seed=None, options=None):
        self.coverage_grid.fill(0)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        # 必须把 action 转换成 (NUM_DRONES, 4) 格式传给父类
        # 假设 action 只是 (4,)
        action_reshaped = np.reshape(action, (1, 4))
        return super().step(action_reshaped)