
import numpy as np
import pybullet as p
from gymnasium import spaces
from ISE_Transformer.envs.BaseAviary import BaseAviary
from ISE_Transformer.utils.enums import DroneModel, Physics
from ISE_Transformer.control.DSLPIDControl import DSLPIDControl

class CoverageAviary(BaseAviary):
    """
    Coverage environment with a LiDAR-like sensor model that supports:

    1) Visibility (field-of-view) coverage: cells are covered if they are within the sensor's
       line-of-sight along emitted rays (occlusion-aware).
    2) Surface point cloud: ray hits generate surface points on walls/obstacles; wall points
       are drawn once (deduped) to avoid visual clutter.
    3) Optional radar-ray visualization toggle via PyBullet UserDebugParameter.

    Notes
    - "Coverage" is defined as *visibility* of ground-projected grid cells along rays.
    - Surface hits are also recorded/visualized, but reward is driven by visibility coverage
      (more consistent with exploration/inspection objectives).
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = True,
                 obstacles: bool = True,
                 # Map/grid parameters
                 rows: int = 50,
                 cols: int = 50,
                 grid_res: float = 0.5,
                 # Sensor parameters
                 radar_radius: float = 8.0,
                 num_rays: int = 180,
                 # Sampling parameters for visibility along rays
                 sample_step: float = None,  # default: grid_res
                 # Visualization
                 point_size_ground: float = 5,
                 point_size_wall: float = 7,
                 # Visualization performance controls
                 viz_rays_every: int = 3,          # update rays every N env steps
                 viz_points_every: int = 5,        # redraw ground points every N env steps
                 viz_max_rays: int = 240,          # limit number of rays drawn (visual only)
                 ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.grid_res = float(grid_res)
        self.map_size = self.rows * self.cols

        # Viz performance knobs
        self.viz_rays_every = int(max(1, viz_rays_every))
        self.viz_points_every = int(max(1, viz_points_every))
        self.viz_max_rays = int(max(1, viz_max_rays))
        self._viz_step = 0

        # Cached debug item ids for efficient updates (avoid per-step reallocation)
        self._radar_line_ids = []
        self._ground_points_item = None
        self._wall_points_item = None
        self._wall_pts_store = []

        # World bounds centered roughly around origin (same as original file assumptions)
        half_x = (self.cols * self.grid_res) / 2.0
        half_y = (self.rows * self.grid_res) / 2.0
        self.grid_bounds = ((-half_x, half_x), (-half_y, half_y))

        # Coverage grid: 0=uncovered, 1=covered (visibility-driven)
        self.coverage_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Visualization bookkeeping
        self.coverage_debug_items = []          # ground coverage points (batched per step)
        self.radar_debug_items = []             # ray lines (per step, cleared/redrawn)
        self.wall_debug_items = []              # wall points (persistent)
        self.wall_marked_cells = set()          # dedupe by (r,c)

        # Sensor params
        self.radar_radius = float(radar_radius)
        self.num_rays = int(num_rays)
        self.sample_step = float(sample_step) if sample_step is not None else self.grid_res

        # Ray direction precompute (horizontal scan)
        self.ray_angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        self.ray_vecs = np.stack([np.cos(self.ray_angles), np.sin(self.ray_angles)], axis=1)

        # Visual sizes
        self.point_size_ground = float(point_size_ground)
        self.point_size_wall = float(point_size_wall)

        # High-level position controller for macro actions
        self._pos_ctrl = DSLPIDControl(drone_model=drone_model)


        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         obstacles=obstacles)

        # Add a GUI toggle for ray visualization
        self._radar_toggle_param = None
        if self.GUI:
            try:
                self._radar_toggle_param = p.addUserDebugParameter("Show Radar", 0, 1, 1, physicsClientId=self.CLIENT)
            except Exception:
                self._radar_toggle_param = None

    def compute_scan_at_pos(self, pos):
        """
        在指定位置进行一次静态覆盖扫描。

        参数:
        pos: [x, y, z] 无人机目标位置

        返回:
        obs: 当前观测值
        reward: 本次扫描新覆盖的面积
        terminated: 是否发生碰撞或越界
        truncated: 始终为 False
        info: 环境信息
        """
        # 1. 物理瞬移：直接将无人机放置到目标点
        # 保持四元数为 [0,0,0,1] 即水平姿态
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            pos,
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )

        # 2. 强制同步状态：更新类内部的 self.pos, self.quat 等变量
        self._updateAndStoreKinematicInformation()

        # 3. 执行覆盖计算：调用类已有的 _computeReward 逻辑
        # 该逻辑会处理：rayTestBatch、更新 coverage_grid、更新 debug 可视化
        new_coverage_reward = self._computeReward()

        # 4. 碰撞检测 (可选但建议)：防止轨迹穿过障碍物
        terminated = False
        contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        if len(contact_points) > 0:
            terminated = True

        # 5. 准备返回值
        obs = self._computeObs()
        info = self._computeInfo()
        truncated = False

        return obs, new_coverage_reward, terminated, truncated, info
    # =========================================================
    #  Gym API: spaces
    # =========================================================
    def _actionSpace(self):
        # Reuse BaseAviary's expected action space type when in velocity control; keep minimal for compatibility.
        # Many gym-pybullet-drones examples use 4D actions: thrust/roll/pitch/yaw or velocity/yaw-rate depending on ctrl.
        # Here we expose a generic continuous box; your controller in BaseAviary will interpret it.
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        # Observation definition is handled by BaseAviary; keep placeholder to satisfy gymnasium wrappers if needed.
        return spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    # =========================================================
    #  Coordinate helpers
    # =========================================================

    def _computeObs(self):
        """Return observation: drone state vector + flattened coverage grid."""
        drone_state = self._getDroneStateVector(0)
        flat_map = self.coverage_grid.flatten().astype(np.float32)
        return np.concatenate([drone_state, flat_map])

    def _preprocessAction(self, action):
        """Convert action to motor RPMs. Here we assume action is RPM-like."""
        return np.clip(action, 0, self.MAX_RPM)

    def _computeTerminated(self):
        """Termination (Gymnasium 'terminated'): keep False; external scripts can stop on collision."""
        return False

    def _computeTruncated(self):
        """Truncation (Gymnasium 'truncated'): keep False; external scripts can time-limit."""
        return False
    def _pos_to_grid(self, x: float, y: float):
        """World (x,y) -> (c,r) grid index; returns (-1,-1) if out of bounds."""
        x_min, x_max = self.grid_bounds[0]
        y_min, y_max = self.grid_bounds[1]
        if x < x_min or x >= x_max or y < y_min or y >= y_max:
            return -1, -1
        c = int((x - x_min) / self.grid_res)
        r = int((y - y_min) / self.grid_res)
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return -1, -1
        return c, r

    def _grid_to_pos(self, c: int, r: int):
        """Grid (c,r) -> world center position (x,y) at cell center."""
        x = self.grid_bounds[0][0] + c * self.grid_res + self.grid_res / 2.0
        y = self.grid_bounds[1][0] + r * self.grid_res + self.grid_res / 2.0
        return x, y

    # =========================================================
    #  Visualization helpers
    # =========================================================
    def _clear_radar_visuals(self):
        ids = list(getattr(self, "_radar_line_ids", []))
        for item_id in ids:
            try:
                p.removeUserDebugItem(item_id, physicsClientId=self.CLIENT)
            except Exception:
                pass
        self._radar_line_ids = []
        self.radar_debug_items = []

    def _clear_coverage_visuals(self):
        if getattr(self, "_ground_points_item", None) is not None:
            try:
                p.removeUserDebugItem(self._ground_points_item, physicsClientId=self.CLIENT)
            except Exception:
                pass
        self._ground_points_item = None
        self.coverage_debug_items = []

    def _clear_wall_visuals(self):
        if getattr(self, "_wall_points_item", None) is not None:
            try:
                p.removeUserDebugItem(self._wall_points_item, physicsClientId=self.CLIENT)
            except Exception:
                pass
        self._wall_points_item = None
        self.wall_debug_items = []
        self._wall_pts_store = []
        self.wall_marked_cells = set()

    def _redraw_full_coverage(self):
        """Redraw the entire covered ground grid as a SINGLE debug point-cloud item."""
        if getattr(self, "_ground_points_item", None) is not None:
            try:
                p.removeUserDebugItem(self._ground_points_item, physicsClientId=self.CLIENT)
            except Exception:
                pass
            self._ground_points_item = None

        rs, cs = np.where(self.coverage_grid == 1)
        if len(rs) == 0:
            return

        pts = []
        cols = []
        for r, c in zip(rs, cs):
            x, y = self._grid_to_pos(int(c), int(r))
            pts.append([x, y, 0.05])
            cols.append([0, 0, 1])

        if pts:
            try:
                self._ground_points_item = p.addUserDebugPoints(
                    pts, cols, pointSize=self.point_size_ground, physicsClientId=self.CLIENT
                )
            except Exception:
                self._ground_points_item = None



    # =========================================================
    #  Reward: Visibility coverage with occlusion-aware rays + surface point cloud
    # =========================================================
    def _computeReward(self):
        """
        Reward = newly visible ground-projected cells * (grid_res^2)

        Also updates:
        - Ground blue points for newly visible cells (batched)
        - Wall/obstacle blue points at 3D hit positions (only once per cell)
        - Optional ray visualization (toggle)
        """
        pos = self.pos[0]
        if pos[2] < 0.1:
            return 0.0

        # 1) Build rays (horizontal scan)
        ray_froms = np.tile(pos, (self.num_rays, 1))
        ray_tos = np.zeros((self.num_rays, 3), dtype=np.float32)
        ray_tos[:, 0] = pos[0] + self.radar_radius * self.ray_vecs[:, 0]
        ray_tos[:, 1] = pos[1] + self.radar_radius * self.ray_vecs[:, 1]
        ray_tos[:, 2] = pos[2]

        results = p.rayTestBatch(ray_froms, ray_tos, physicsClientId=self.CLIENT)

        # Optional: draw rays (optimized: reuse debug line ids, decimate updates)
        show_radar = 1
        if self._radar_toggle_param is not None:
            try:
                show_radar = int(p.readUserDebugParameter(self._radar_toggle_param, physicsClientId=self.CLIENT))
            except Exception:
                show_radar = 1

        if self.GUI and show_radar and (self._viz_step % self.viz_rays_every == 0):
            # Only draw a subset for performance (visual only)
            n_vis = min(int(self.viz_max_rays), int(self.num_rays))
            stride = max(1, int(np.ceil(self.num_rays / n_vis)))
            vis_indices = list(range(0, self.num_rays, stride))

            # Ensure cache length matches
            if len(self._radar_line_ids) != len(vis_indices):
                self._clear_radar_visuals()

            for k, i_ray in enumerate(vis_indices):
                res = results[i_ray]
                end_pt = res[3] if res[0] != -1 else ray_tos[i_ray].tolist()
                try:
                    if self._radar_line_ids and len(self._radar_line_ids) == len(vis_indices):
                        line_id = p.addUserDebugLine(
                            lineFromXYZ=pos.tolist(),
                            lineToXYZ=end_pt,
                            lineColorRGB=[1, 0, 0],
                            lineWidth=1,
                            lifeTime=0,
                            replaceItemUniqueId=self._radar_line_ids[k],
                            physicsClientId=self.CLIENT,
                        )
                        self._radar_line_ids[k] = line_id
                    else:
                        line_id = p.addUserDebugLine(
                            lineFromXYZ=pos.tolist(),
                            lineToXYZ=end_pt,
                            lineColorRGB=[1, 0, 0],
                            lineWidth=1,
                            lifeTime=0,
                            physicsClientId=self.CLIENT,
                        )
                        self._radar_line_ids.append(line_id)
                except Exception:
                    pass
        elif self.GUI and (not show_radar):
            if getattr(self, "_radar_line_ids", []):
                self._clear_radar_visuals()

        # Backward-compat list
        self.radar_debug_items = []

        # Precompute origin xy for faster per-ray
        ox, oy = float(pos[0]), float(pos[1])
        # Visibility mask for this step (ground-projected, occlusion-aware)
        visible_mask = np.zeros_like(self.coverage_grid, dtype=np.uint8)

        # Per-step hit points (cell -> 3D hit position) for wall/obstacle surface marking
        hit_cell_points = {}


        for i, res in enumerate(results):
            # Determine the segment end (occlusion-aware)
            if res[0] != -1:
                end = res[3]  # hit position (x,y,z)
                ex, ey = float(end[0]), float(end[1])
                # record hit cell + 3D point
                c_hit, r_hit = self._pos_to_grid(ex, ey)
                if r_hit != -1:
                    # keep the closest hit point for this cell (rare collisions)
                    hit_cell_points[(r_hit, c_hit)] = [ex, ey, float(end[2])]
            else:
                ex, ey = float(ray_tos[i, 0]), float(ray_tos[i, 1])

            # Trace visibility along the ray segment (origin -> end)
            dx = ex - ox
            dy = ey - oy
            seg_len = float(np.hypot(dx, dy))
            if seg_len <= 1e-6:
                continue

            steps = max(1, int(seg_len / self.sample_step))
            # sample at fixed distance increments, including near-end
            for s in range(steps + 1):
                d = min(seg_len, s * self.sample_step)
                px = ox + (d / seg_len) * dx
                py = oy + (d / seg_len) * dy
                c, r = self._pos_to_grid(px, py)
                if r == -1:
                    continue
                visible_mask[r, c] = 1

        # 4) Update coverage grid and compute reward
        newly_visible_mask = (visible_mask == 1) & (self.coverage_grid == 0)
        new_count = int(np.sum(newly_visible_mask))
        if new_count > 0:
            self.coverage_grid[newly_visible_mask] = 1

        # 5) Draw ground coverage (optimized: redraw full set periodically)
        if self.GUI and (new_count > 0) and (self._viz_step % self.viz_points_every == 0):
            self._redraw_full_coverage()


        # 6) Draw wall/obstacle hit points ONCE (deduped) as a SINGLE item
        if self.GUI and hit_cell_points:
            new_wall_pts = []
            for (r, c), hit_pos in hit_cell_points.items():
                key = (int(r), int(c))
                if key in self.wall_marked_cells:
                    continue
                self.wall_marked_cells.add(key)
                new_wall_pts.append([hit_pos[0], hit_pos[1], hit_pos[2] + 0.01])

            if new_wall_pts:
                # Accumulate then redraw a single point-cloud item (updates are infrequent)
                self._wall_pts_store.extend(new_wall_pts)
                if getattr(self, "_wall_points_item", None) is not None:
                    try:
                        p.removeUserDebugItem(self._wall_points_item, physicsClientId=self.CLIENT)
                    except Exception:
                        pass
                    self._wall_points_item = None

                try:
                    cols = [[0, 0, 1] for _ in range(len(self._wall_pts_store))]
                    self._wall_points_item = p.addUserDebugPoints(
                        self._wall_pts_store, cols, pointSize=self.point_size_wall, physicsClientId=self.CLIENT
                    )
                except Exception:
                    self._wall_points_item = None


        # 7) Provide some useful info for debugging/learning curves
        # BaseAviary's step() typically calls _computeInfo() separately, but keeping reward pure is fine.
        self._viz_step += 1
        return new_count * (self.grid_res ** 2)

    # =========================================================
    #  Info (optional): expose coverage ratio, etc.
    # =========================================================
    def _computeInfo(self):
        covered = int(np.sum(self.coverage_grid))
        return {
            "covered_cells": covered,
            "coverage_ratio": covered / float(self.map_size),
        }

    # =========================================================
    #  Obstacles/world generation (reuse a simple world)
    # =========================================================
    def _addObstacles(self):
        """Create boundary walls and some random box obstacles (static)."""
        client = self.CLIENT
        x_min, x_max = self.grid_bounds[0]
        y_min, y_max = self.grid_bounds[1]

        # Boundary walls (thin boxes)
        thickness = 0.1
        height = 2.0
        # Left wall
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[thickness, (y_max - y_min) / 2, height / 2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[thickness, (y_max - y_min) / 2, height / 2], rgbaColor=[0.6, 0.6, 0.6, 1])
        p.createMultiBody(0, col, vis, [x_min, 0, height / 2], physicsClientId=client)
        # Right wall
        p.createMultiBody(0, col, vis, [x_max, 0, height / 2], physicsClientId=client)
        # Bottom wall
        col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[(x_max - x_min) / 2, thickness, height / 2])
        vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[(x_max - x_min) / 2, thickness, height / 2], rgbaColor=[0.6, 0.6, 0.6, 1])
        p.createMultiBody(0, col2, vis2, [0, y_min, height / 2], physicsClientId=client)
        # Top wall
        p.createMultiBody(0, col2, vis2, [0, y_max, height / 2], physicsClientId=client)

        # Random obstacles
        rng = np.random.RandomState(101)
        for _ in range(8):
            ox = rng.uniform(x_min + 1.0, x_max - 1.0)
            oy = rng.uniform(y_min + 1.0, y_max - 1.0)
            if abs(ox) < 1.5 and abs(oy) < 1.5:
                continue
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 1.0])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 1.0], rgbaColor=[0.4, 0.4, 0.4, 1])
            p.createMultiBody(0, col, vis, [ox, oy, 1.0], physicsClientId=client)

    # =========================================================
    #  Snapshot utilities (optional)
    # =========================================================
    def take_snapshot(self):
        """Snapshot physical state + coverage grid (useful for planning/rollback)."""
        snap_id = p.saveState(physicsClientId=self.CLIENT)
        return {"phys_id": snap_id, "grid": self.coverage_grid.copy()}

    def restore_snapshot(self, snapshot):
        """Restore snapshot and redraw coverage points."""
        if snapshot is None:
            return
        p.restoreState(snapshot["phys_id"], physicsClientId=self.CLIENT)
        self.coverage_grid = snapshot["grid"].copy()


        # self._updateAndStoreKinematicInformation()
        self.pos[0] = snapshot['pos']
        self._clear_coverage_visuals()
        self._redraw_full_coverage()
        self.reset_macro_controller()

    def remove_snapshot(self, snapshot):
        if snapshot is None:
            return
        try:
            p.removeState(snapshot["phys_id"], physicsClientId=self.CLIENT)
        except Exception:
            pass

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

        # 2. 逻辑数据 (必须深拷贝!)
        grid_copy = self.coverage_grid.copy()
        pos = self.pos[0].copy()

        return {
            'phys_id': phys_id,
            'grid': grid_copy,
            'pos': pos
        }

    # =========================================================
    #  Macro-action: 27-neighbor grid move (fixed landing position)
    # =========================================================
    def _action27_to_delta(self, a: int):
        """
        Map a in [0,26] to (dx,dy,dz) where each is in {-1,0,1}.
        Order: lexicographic over dx,dy,dz.
        """
        if not (0 <= int(a) <= 26):
            raise ValueError(f"action27 out of range: {a}")
        a = int(a)
        dx = (a // 9) - 1
        dy = ((a % 9) // 3) - 1
        dz = (a % 3) - 1
        return dx, dy, dz

    def _snap_to_grid_center_3d(self, x, y, z, cell_xy=None, cell_z=None):
        """Snap (x,y,z) to the nearest grid-center in xy and (optional) in z."""
        cell_xy = float(cell_xy if cell_xy is not None else self.grid_res)
        cell_z = float(cell_z if cell_z is not None else self.grid_res)

        # xy bounds use your grid_bounds
        x_min, _ = self.grid_bounds[0]
        y_min, _ = self.grid_bounds[1]

        gx = int(np.clip(np.round((x - x_min - cell_xy / 2) / cell_xy), 0, self.cols - 1))
        gy = int(np.clip(np.round((y - y_min - cell_xy / 2) / cell_xy), 0, self.rows - 1))

        x_c = x_min + gx * cell_xy + cell_xy / 2
        y_c = y_min + gy * cell_xy + cell_xy / 2

        # z：可选离散（如果你不想离散高度，把下面两行改成 z_c = z）
        gz = int(np.round(z / cell_z))
        z_c = max(0.1, gz * cell_z)  # 防止贴地

        return np.array([x_c, y_c, z_c], dtype=np.float32)

    def macro_step(self,
                   a27: int,
                   cell_xy: float = None,
                   cell_z: float = None,
                   max_inner_steps: int = 60,
                   pos_tol: float = 0.12,
                   yaw_tol: float = 0.35,
                   target_yaw: float = 0.0):
        """
        Execute one macro-action: move to a neighboring grid center (27 actions).

        Returns Gymnasium 5-tuple:
            obs, reward, terminated, truncated, info

        Key properties:
        - Deterministic landing: same discrete state + same action -> same target grid center.
        - Closed-loop control using DSLPID to track the target.
        - Occlusion-aware visibility coverage + wall points are still handled by _computeReward().
        """
        # 0) Decode action to delta
        dx, dy, dz = self._action27_to_delta(a27)

        # 1) Current position snapped to grid center (so target is stable)
        cur = self.pos[0]
        cur_snap = self._snap_to_grid_center_3d(cur[0], cur[1], cur[2], cell_xy=cell_xy, cell_z=cell_z)

        # 2) Target grid center
        step_xy = float(cell_xy if cell_xy is not None else self.grid_res)
        step_z = float(cell_z if cell_z is not None else self.grid_res)

        target = cur_snap + np.array([dx * step_xy, dy * step_xy, dz * step_z], dtype=np.float32)

        # Keep target inside XY bounds
        x_min, x_max = self.grid_bounds[0]
        y_min, y_max = self.grid_bounds[1]
        target[0] = float(np.clip(target[0], x_min + step_xy / 2, x_max - step_xy / 2))
        target[1] = float(np.clip(target[1], y_min + step_xy / 2, y_max - step_xy / 2))
        target[2] = float(max(0.2, target[2]))  # 最低飞行高度可按需改

        # 3) Inner-loop control to reach target
        terminated = False
        truncated = False
        total_reward = 0.0

        # If you want “one macro step = one reward eval only”, set this to False.
        accumulate_reward = True

        for k in range(int(max_inner_steps)):
            state = self._getDroneStateVector(0)

            # State convention in gym-pybullet-drones:
            # state[0:3]=pos, state[3:7]=quat, state[7:10]=rpy, state[10:13]=vel, ...
            # (Different versions vary; DSLPIDControl handles standard vector from _getDroneStateVector.)
            rpm, _, _ = self._pos_ctrl.computeControlFromState(
                control_timestep=self.CTRL_TIMESTEP,
                state=state,
                target_pos=target,
                target_rpy=np.array([0.0, 0.0, float(target_yaw)]),
                target_vel=np.zeros(3),
                target_rpy_rates=np.zeros(3),
            )

            # BaseAviary.step() expects an action; in RPM-control mode passing rpm is correct.
            obs, r, term, trunc, info = super().step(rpm)

            if accumulate_reward:
                total_reward += float(r)
            else:
                total_reward = float(r)

            # reach check
            p_now = self.pos[0]
            dist = float(np.linalg.norm(np.array(p_now) - target))
            # yaw error (rough): use rpy if available in state (common)
            try:
                yaw_now = float(state[9])  # some versions: rpy[2] at index 9
            except Exception:
                yaw_now = 0.0
            yaw_err = float(abs((yaw_now - target_yaw + np.pi) % (2 * np.pi) - np.pi))

            if dist <= float(pos_tol) and yaw_err <= float(yaw_tol):
                break

            terminated = bool(term)
            truncated = bool(trunc)
            if terminated or truncated:
                break

        # If didn’t reach within inner steps, mark truncated (macro-action timeout)
        p_now = self.pos[0]
        if float(np.linalg.norm(np.array(p_now) - target)) > float(pos_tol):
            truncated = True

        # One macro step returns the latest obs/info; reward is accumulated over inner steps.
        info = dict(info) if isinstance(info, dict) else {}
        info.update({
            "macro_target": target.tolist(),
            "macro_action": int(a27),
            "macro_reached": (not truncated),
        })

        return obs, float(total_reward), bool(terminated), bool(truncated), info

    def reset_macro_controller(self):
        """Reset high-level controller state to make rollouts repeatable."""
        try:
            self._pos_ctrl = DSLPIDControl(drone_model=self.DRONE_MODEL)
        except Exception:
            # If your project stores drone_model differently:
            self._pos_ctrl = DSLPIDControl(drone_model=self.drone_model)