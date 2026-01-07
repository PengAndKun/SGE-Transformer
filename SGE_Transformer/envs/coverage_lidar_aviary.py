import numpy as np
import pybullet as p
from gymnasium import spaces
from SGE_Transformer.envs.BaseAviary import BaseAviary
from SGE_Transformer.utils.enums import DroneModel, Physics

# 尝试导入 OpenCV，用于多边形填充
try:
    import cv2
except ImportError:
    print("[Error] OpenCV not installed. Please run: pip install opencv-python")
    cv2 = None


class CoverageAviary(BaseAviary):
    def __init__(self,
                 grid_bounds=((-5, 5), (-5, 5)),
                 grid_res=0.2,
                 fov_angle=np.deg2rad(45),
                 radar_radius=3.0,
                 num_rays=36,
                 drone_model=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq=240,
                 ctrl_freq=30,
                 gui=False,
                 obstacles=True,
                 user_debug_gui=True,
                 **kwargs):

        # 1. 覆盖网格初始化
        self.grid_bounds = grid_bounds
        self.grid_res = grid_res
        self.fov_angle = fov_angle
        self.radar_radius = radar_radius
        self.num_rays = num_rays

        self.rows = int((grid_bounds[0][1] - grid_bounds[0][0]) / grid_res)
        self.cols = int((grid_bounds[1][1] - grid_bounds[1][0]) / grid_res)
        self.map_size = self.rows * self.cols

        # 核心状态变量：覆盖栅格 (0=未覆盖, 1=已覆盖)
        self.coverage_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # 视觉管理：存储 PyBullet UserDebugItem 的 ID
        self.coverage_debug_items = []

        # 10_e. 射线向量预计算 (避免每步重复计算 sin/cos)
        self.ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        self.ray_vecs = np.stack([np.cos(self.ray_angles), np.sin(self.ray_angles)], axis=1)

        super().__init__(drone_model=drone_model, num_drones=1, initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys, physics=physics, pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq, gui=gui, obstacles=obstacles, user_debug_gui=user_debug_gui, **kwargs)

    def _actionSpace(self):
        act_lower_bound = np.array([0., 0., 0., 0.])
        act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        # 观测空间：无人机状态 (20维) + 扁平化的地图
        phys_dim = 20
        return spaces.Box(low=0, high=1, shape=(phys_dim + self.map_size,), dtype=np.float32)

    def _computeObs(self):
        drone_state = self._getDroneStateVector(0)
        flat_map = self.coverage_grid.flatten().astype(np.float32)
        return np.concatenate([drone_state, flat_map])

    def _preprocessAction(self, action):
        return np.clip(action, 0, self.MAX_RPM)

    # =========================================================
    #  核心功能：碰撞检测辅助函数 (供外部脚本调用)
    # =========================================================
    def check_collision(self):
        """
        检测无人机是否发生碰撞或处于非法状态。
        返回: Boolean (True 表示撞了/翻了/出界了)
        """
        # 1. 物理接触检测 (撞墙/撞地)
        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        if len(contact_points) > 0:
            return True

        # 10_e. 姿态检测 (翻车检测)
        r, p_angle, y = self.rpy[0]
        if abs(r) > 1.5 or abs(p_angle) > 1.5:  # 倾角过大
            return True

        # 3. 高度/边界检测
        z_height = self.pos[0][2]
        if z_height < 0.05 or z_height > 2.5:  # 掉地底下或飞太高
            return True

        return False

    # =========================================================
    #  核心功能：视觉管理 (蓝点绘制)
    # =========================================================
    def _clear_coverage_visuals(self):
        """清除屏幕上所有的覆盖标记 (蓝点)"""
        if self.GUI:
            for item_id in self.coverage_debug_items:
                p.removeUserDebugItem(item_id, physicsClientId=self.CLIENT)
            self.coverage_debug_items.clear()

    def _redraw_full_coverage(self):
        """根据当前的 coverage_grid 重新绘制所有蓝点 (用于回放/恢复存档时)"""
        if not self.GUI: return

        # 找到所有值为1的坐标
        rows, cols = np.where(self.coverage_grid == 1)

        if len(rows) > 0:
            points = []
            colors = []
            for r, c in zip(rows, cols):
                x, y = self._grid_to_pos(c, r)
                points.append([x, y, 0.05])  # 稍微浮起一点点，防止与地面z-fighting
                colors.append([0, 0, 1])  # 纯蓝

            # 批量绘制以提高性能
            item_id = p.addUserDebugPoints(
                pointPositions=points,
                pointColorsRGB=colors,
                pointSize=5,
                physicsClientId=self.CLIENT
            )
            self.coverage_debug_items.append(item_id)

    # =========================================================
    #  SGE 核心：快照系统 (Snapshot)
    # =========================================================
    def get_snapshot(self):
        """
        保存当前环境的完整状态。
        包含: 物理引擎状态 (PyBullet) + 逻辑状态 (Coverage Grid)
        """
        # 1. 物理快照
        phys_id = p.saveState(physicsClientId=self.CLIENT)

        # 10_e. 逻辑数据 (必须深拷贝!)
        grid_copy = self.coverage_grid.copy()
        pos = self.pos[0]

        return {
            'phys_id': phys_id,
            'grid': grid_copy,
            'pos': pos
        }

    def restore_snapshot(self, snapshot):
        """
        恢复到指定的快照状态。
        自动处理: 物理重置 + 变量重置 + 画面重绘
        """
        if snapshot is None: return

        # 1. 恢复物理世界
        p.restoreState(snapshot['phys_id'], physicsClientId=self.CLIENT)

        # 10_e. 恢复覆盖数据
        self.coverage_grid = snapshot['grid'].copy()

        # 3. 告诉 BaseAviary 更新内部变量 (pos, vel, rpy 等)
        self._updateAndStoreKinematicInformation()

        # 4. 视觉同步: 删掉当前屏幕上的所有点，按恢复后的 Grid 重画
        self._clear_coverage_visuals()
        self._redraw_full_coverage()

    def remove_snapshot(self, snapshot):
        """释放内存"""
        if snapshot and 'phys_id' in snapshot:
            p.removeState(snapshot['phys_id'], physicsClientId=self.CLIENT)

    # =========================================================
    #  核心功能：奖励计算 (射线覆盖逻辑)
    # =========================================================
    def _computeReward(self):
        """
        计算单步奖励：
        Reward = 新增覆盖面积 (平方米)
        同时负责更新 GUI 上的蓝点
        """
        pos = self.pos[0]
        # 如果还没起飞(高度过低)，不计算覆盖
        if pos[2] < 0.1: return 0.0

        # --- 1. 射线检测 (Ray Casting) ---
        ray_froms = np.tile(pos, (self.num_rays, 1))
        ray_tos = np.zeros((self.num_rays, 3))
        # 计算射线终点
        ray_tos[:, 0] = pos[0] + self.radar_radius * self.ray_vecs[:, 0]
        ray_tos[:, 1] = pos[1] + self.radar_radius * self.ray_vecs[:, 1]
        ray_tos[:, 2] = pos[2]

        # 批量射线检测
        results = p.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.CLIENT)

        polygon_points = []
        for i, res in enumerate(results):
            hit_x, hit_y = ray_tos[i, 0], ray_tos[i, 1]

            # 如果撞到墙 (res[0] != -1)，截断射线
            if res[0] != -1:
                hit_fraction = res[2]
                hit_x = pos[0] + hit_fraction * self.radar_radius * self.ray_vecs[i, 0]
                hit_y = pos[1] + hit_fraction * self.radar_radius * self.ray_vecs[i, 1]

            # 转为网格坐标
            c, r = self._pos_to_grid(hit_x, hit_y)
            polygon_points.append([c, r])

        polygon_points = np.array(polygon_points, dtype=np.int32)

        # --- 10_e. 计算覆盖增量 (利用 OpenCV 填充多边形) ---
        current_scan_mask = np.zeros_like(self.coverage_grid, dtype=np.uint8)
        if len(polygon_points) > 0 and cv2 is not None:
            cv2.fillPoly(current_scan_mask, [polygon_points], 1)

        # 找出: 本次扫描到 且 之前没覆盖 的区域
        newly_covered_mask = (current_scan_mask == 1) & (self.coverage_grid == 0)

        # --- 3. 可视化增量 (仅在 GUI 模式下) ---
        if self.GUI:
            new_rows, new_cols = np.where(newly_covered_mask)
            if len(new_rows) > 0:
                points_to_draw = []
                colors_to_draw = []
                for r, c in zip(new_rows, new_cols):
                    x, y = self._grid_to_pos(c, r)
                    points_to_draw.append([x, y, 0.05])
                    colors_to_draw.append([0, 0, 1])

                if points_to_draw:
                    item_id = p.addUserDebugPoints(
                        pointPositions=points_to_draw,
                        pointColorsRGB=colors_to_draw,
                        pointSize=5,
                        physicsClientId=self.CLIENT
                    )
                    self.coverage_debug_items.append(item_id)

        # --- 4. 更新状态与返回奖励 ---
        new_count = np.sum(newly_covered_mask)
        self.coverage_grid[newly_covered_mask] = 1

        # 返回单纯的覆盖面积增量，不做负分惩罚
        return new_count * (self.grid_res ** 2)

    def _grid_to_pos(self, c, r):
        """网格索引 -> 物理中心坐标"""
        x = self.grid_bounds[0][0] + c * self.grid_res + self.grid_res / 2
        y = self.grid_bounds[1][0] + r * self.grid_res + self.grid_res / 2
        return x, y

    def _pos_to_grid(self, x, y):
        """物理坐标 -> 网格索引"""
        c = int((x - self.grid_bounds[0][0]) / self.grid_res)
        r = int((y - self.grid_bounds[1][0]) / self.grid_res)
        return np.clip(c, 0, self.cols - 1), np.clip(r, 0, self.rows - 1)

    def _addObstacles(self):
        """生成房间和障碍物"""
        client = self.CLIENT
        x_min, x_max = self.grid_bounds[0]
        y_min, y_max = self.grid_bounds[1]
        h = 2.0
        t = 0.5

        # 四周墙壁
        walls = [
            ([x_min - t, 0, h / 2], [t, (y_max - y_min) / 2 + t, h / 2]),
            ([x_max + t, 0, h / 2], [t, (y_max - y_min) / 2 + t, h / 2]),
            ([0, y_min - t, h / 2], [(x_max - x_min) / 2 + t, t, h / 2]),
            ([0, y_max + t, h / 2], [(x_max - x_min) / 2 + t, t, h / 2])
        ]
        for pos, ext in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=[0.7, 0.7, 0.7, 1])
            p.createMultiBody(0, col, vis, pos, physicsClientId=client)

        # 随机障碍物
        rng = np.random.RandomState(101)

        for _ in range(6):
            ox = rng.uniform(x_min + 1, x_max - 1)
            oy = rng.uniform(y_min + 1, y_max - 1)

            if abs(ox) < 1.5 and abs(oy) < 1.5: continue  # 避开起点

            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 1.0])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 1.0], rgbaColor=[0.4, 0.4, 0.4, 1])
            p.createMultiBody(0, col, vis, [ox, oy, 1.0], physicsClientId=client)

    # 兼容 Gym 接口的空实现
    def _computeTerminated(self):
        # 我们在外部控制终止条件 (碰撞检测)
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        # 可以在这里返回当前覆盖率
        covered_count = np.sum(self.coverage_grid)
        return {"coverage_ratio": covered_count / self.map_size}

    def reset(self, seed=None, options=None):
        self._clear_coverage_visuals()  # 重置时必须清空所有蓝点
        self.coverage_grid.fill(0)  # 重置覆盖数据
        return super().reset(seed=seed, options=options)