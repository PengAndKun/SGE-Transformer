# (首先确保您已安装必要的库: pip install numpy matplotlib seaborn)
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from dataclasses import dataclass


# --- Configuration Management---
@dataclass
class GESConfig:
    K: int = 5  # 总宏观迭代次数 Total macro iteration times
    A: int = 8  # 最大存档数量 Maximum number of archives
    L: int = 10  # 每次探索的步长 Step size for each exploration
    EXPLORATION_TIMES: int = 1000  # 每个存档内的探索循环次数 The number of exploration cycles in each archive
    NUM_SIMULATIONS: int = 10  # 估算期望增益的模拟次数 Estimate the number of simulations for expected gain
    CANDIDATE_POOL_SIZE: int = 20  # 选择最佳节点时的候选池大小 Candidate pool size when selecting the best node


class GridWorldEnv:
    def __init__(self):
        self.grid = np.array([
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ])
        self.height, self.width = self.grid.shape
        self.fov_pattern = [(0, 0), (-1, 0), (-1, 1), (0, 1)]
        self.start_state = (6, 8)
        self.agent_pos = self.start_state

    def reset(self, state=None):
        self.agent_pos = state if state else self.start_state; return self.agent_pos

    def get_state(self):
        return self.agent_pos

    def step(self, action):
        y, x = self.agent_pos
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < self.height - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.width - 1:
            x += 1
        if self.grid[y, x] == 0: self.agent_pos = (y, x)
        return self.agent_pos, 0, False, {}

    def get_field_of_view(self, position):
        y, x = position
        visible_cells = set()
        for dy, dx in self.fov_pattern:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny, nx] == 0:
                visible_cells.add((ny, nx))
        return visible_cells


class ExplorationNode:
    def __init__(self, key, path_states, path_actions, observed_set, node_type='standard'):
        self.key = key;
        self.path_states = path_states;
        self.path_actions = path_actions
        self.observed_set = observed_set;
        self.value = len(observed_set)
        self.times_selected = 0;
        self.node_type = node_type

    def __repr__(self):
        return f"Node(key={self.key}, value={self.value}, selected={self.times_selected}, type='{self.node_type}')"


def select_archive(archive):
    if not archive: return None
    min_times_selected = min(data['times_selected'] for data in archive.values())
    least_visited_archives = [id for id, data in archive.items() if data['times_selected'] == min_times_selected]
    return random.choice(least_visited_archives)


def select_elite_cells_from_cells_pool(cells_pool):
    if not cells_pool: return [], 0
    max_observed_count = max(node.value for node in cells_pool.values())
    elite_cell_keys = [key for key, node in cells_pool.items() if node.value == max_observed_count]
    return elite_cell_keys, max_observed_count


def estimate_expected_marginal_gain(env, start_state, base_observed_set, L, num_simulations):
    total_marginal_gain = 0
    for _ in range(num_simulations):
        env.reset(state=start_state)
        current_observed = base_observed_set.copy()
        for _ in range(L):
            action = random.randint(0, 3)
            next_state, _, _, _ = env.step(action)
            current_observed.update(env.get_field_of_view(next_state))
        marginal_gain = len(current_observed) - len(base_observed_set)
        total_marginal_gain += marginal_gain
    return total_marginal_gain / num_simulations


def select_best_cell_from_pool(env, cells_pool, config: GESConfig):
    if not cells_pool: return None, None
    sorted_cells = sorted(cells_pool.items(), key=lambda item: item[1].times_selected)
    num_candidates = min(config.CANDIDATE_POOL_SIZE, len(sorted_cells))
    candidate_cells = sorted_cells[:num_candidates]
    best_cell_key = None
    max_expected_gain = -1
    for cell_key, node in candidate_cells:
        expected_gain = estimate_expected_marginal_gain(env, node.key[1], node.observed_set, config.L,
                                                        config.NUM_SIMULATIONS)
        if expected_gain > max_expected_gain:
            max_expected_gain = expected_gain
            best_cell_key = cell_key
    if best_cell_key is None: best_cell_key = sorted_cells[0][0]
    return best_cell_key, cells_pool[best_cell_key]


def select_greedy_action(env: GridWorldEnv, current_state, current_observed_set):
    action_gains = []
    for action in range(4):
        y, x = current_state
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < env.height - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < env.width - 1:
            x += 1
        next_state = (y, x)
        if env.grid[next_state] == 1 or next_state == current_state:
            action_gains.append(-1);
            continue
        marginal_gain = len(env.get_field_of_view(next_state) - current_observed_set)
        action_gains.append(marginal_gain)
    max_gain = max(action_gains)
    if max_gain <= 0:
        valid_actions = [a for a, g in enumerate(action_gains) if g >= 0]
        return random.choice(valid_actions) if valid_actions else random.randint(0, 3)
    best_actions = [a for a, g in enumerate(action_gains) if g == max_gain]
    return random.choice(best_actions)


# --- 主循环 --- Main Loop
def run_ges_with_fov(env: GridWorldEnv, config: GESConfig):
    initial_state = env.reset()
    archive = {0: {'cells_pool': {}, 'times_selected': 0}}
    optimal_trajectory_archive = {}
    for k in range(config.K):
        archive_id = np.mod(k, config.A)
        cells_pool = archive[archive_id]['cells_pool']
        if  archive_id == 0:
            key = (0, initial_state)
            cells_pool.clear()
            cells_pool[key] = ExplorationNode(key, [initial_state], [], env.get_field_of_view(initial_state))
        print(
            f"\n--- Timestep {k + 1}/{config.K} | Archive {archive_id} ({len(archive[archive_id]['cells_pool'])} nodes) ---")
        archive[archive_id]['times_selected'] += 1

        nodes_added_this_round = 0
        for s in range(config.EXPLORATION_TIMES):
            key_to_explore, start_node = select_best_cell_from_pool(env, cells_pool, config)
            if start_node is None: break
            start_node.times_selected += 1
            path_states = start_node.path_states.copy()
            path_actions = start_node.path_actions.copy()
            observed_states = start_node.observed_set.copy()
            env.reset(state=start_node.key[1])
            current_state = start_node.key[1]
            max_steps = archive_id * config.L + config.L - start_node.key[0]- 1
            for _ in range(max_steps):
                action = select_greedy_action(env, current_state, observed_states)
                next_state, _, _, _ = env.step(action)
                path_actions.append(action)
                path_states.append(next_state)
                observed_states.update(env.get_field_of_view(next_state))
                current_state = next_state
                new_key = (len(path_actions), next_state)
                if new_key not in cells_pool or len(observed_states) > cells_pool[new_key].value:
                    cells_pool[new_key] = ExplorationNode(new_key, path_states.copy(), path_actions.copy(),
                                                          observed_states.copy())
                    nodes_added_this_round += 1
        print(f"Exploration finished. Added {nodes_added_this_round} nodes. Pool size: {len(cells_pool)}")
        elite_keys, max_value = select_elite_cells_from_cells_pool(cells_pool)
        print(f"Found {len(elite_keys)} elite nodes with value {max_value}")
        # 只在最后一个档案时将最优节点加入到最优轨迹档案库中
        if archive_id == config.A - 1:  # 最后一个档案
            for key in elite_keys:
                elite_node = cells_pool[key]
                # 使用唯一键来避免重复 (档案ID + 原键)
                unique_key = (archive_id, key)
                optimal_trajectory_archive[unique_key] = elite_node
            print(
                f"Added {len(elite_keys)} optimal nodes from final archive to trajectory archive. Total archive size: {len(optimal_trajectory_archive)}")
        next_archive_id = archive_id + 1
        # if next_archive_id >= config.A: print("Archive limit reached."); continue
        # new_elite_pool = {}
        # for key in elite_keys:
        #     elite_node = cells_pool[key]
        #     elite_node.times_selected = 0
        #     elite_node.node_type = 'elite'
        #     new_elite_pool[key] = elite_node
        # if next_archive_id not in archive or max_value > archive[next_archive_id].get('start_marginal_gain', -1):
        #     archive[next_archive_id] = {'cells_pool': new_elite_pool, 'times_selected': 0,
        #                                 'start_marginal_gain': max_value}
        #     print(f"Populated Archive {next_archive_id} with {len(elite_keys)} new elite nodes.")
        # elif max_value == archive[next_archive_id].get('start_marginal_gain', -1):
        #     archive[next_archive_id]['cells_pool'].update(new_elite_pool)
        #     print(f"Added {len(new_elite_pool)} more elite nodes to Archive {next_archive_id}.")
        new_elite_pool = {}
        for key in elite_keys:
            elite_node = cells_pool[key]
            elite_node.times_selected = 0
            elite_node.node_type = 'elite'
            new_elite_pool[key] = elite_node
        archive[next_archive_id] = {'cells_pool': new_elite_pool, 'times_selected': 0,
                                    'start_marginal_gain': max_value}
        print(f"Populated Archive {next_archive_id} with {len(elite_keys)} new elite nodes.")
    return archive,optimal_trajectory_archive


def render_optimal_trajectories(env: GridWorldEnv, trajectories: list, final_observed_set: set):
    """
    为每个最优轨迹生成单独的可视化图，并用箭头显示路径方向。
    - 灰色: 墙壁
    - 浅绿色: 被轨迹观测到的区域
    - 蓝色: 轨迹路径
    - 亮绿色: 起点
    - 红色: 终点
    - 箭头: 显示移动方向

    Generate separate visualization for each optimal trajectory with arrows showing path direction.
    """
    if not trajectories:
        print("No trajectories to render.")
        return

    # 为每个轨迹生成单独的图
    for idx, trajectory in enumerate(trajectories):
        plt.figure(figsize=(12, 8))

        # 创建可视化网格
        vis_grid = np.zeros_like(env.grid, dtype=int)

        # 1. 标记墙壁
        vis_grid[env.grid == 1] = 5  # Grey (index 5)

        # 获取轨迹的起点和终点
        start_y, start_x = trajectory[0]
        end_y, end_x = trajectory[-1]

        # 2. 标记观测区域（排除起点和终点）
        for y, x in final_observed_set:
            if env.grid[y, x] == 0 and (y, x) != (start_y, start_x) and (y, x) != (end_y, end_x):
                vis_grid[y, x] = 1  # yellow (index 1)

        # 3. 标记当前轨迹路径（排除起点和终点）
        for y, x in trajectory:
            if (y, x) != (start_y, start_x) and (y, x) != (end_y, end_x):
                vis_grid[y, x] = 3  # Blue (index 3)

        # 4. 最后标记起点和终点（确保它们不被覆盖）
        vis_grid[start_y, start_x] = 4  # Bright Green (Start, index 4)
        vis_grid[end_y, end_x] = 2  # Red (End, index 2)

        # 创建颜色映射
        cmap = ListedColormap(['white', 'yellow', 'red', 'cornflowerblue', 'springgreen', 'grey'])

        # 绘制热力图
        ax = sns.heatmap(vis_grid, cmap=cmap, cbar=False, linewidths=.5, linecolor='lightgray', vmin=0,
                         vmax=5)  # 强制颜色索引范围

        # 5. 添加箭头显示路径方向
        for i in range(len(trajectory) - 1):
            y1, x1 = trajectory[i]
            y2, x2 = trajectory[i + 1]

            # 计算箭头方向
            dy = y2 - y1
            dx = x2 - x1

            # 箭头起点和终点坐标（注意matplotlib的坐标系统）
            arrow_x = x1 + 0.5  # 网格中心
            arrow_y = y1 + 0.5
            arrow_dx = dx * 0.3  # 箭头长度缩放
            arrow_dy = dy * 0.3

            # 绘制箭头
            ax.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy,
                     head_width=0.15, head_length=0.1,
                     fc='white', ec='black', linewidth=1.5, alpha=0.8)

        # 设置标题和显示
        title_msg = f"Optimal Trajectory {idx + 1}/{len(trajectories)}\n" \
                    f"Path Length: {len(trajectory)} states | Observed Cells: {len(final_observed_set)}"
        plt.title(title_msg, fontsize=14, pad=20)

        # 设置坐标轴
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='grey', label='Wall'),
            Patch(facecolor='yellow', label='Observed Area'),
            Patch(facecolor='cornflowerblue', label='Path'),
            Patch(facecolor='springgreen', label='Start'),
            Patch(facecolor='red', label='End'),
        ]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()

    print(f"\nGenerated {len(trajectories)} trajectory visualization(s)")


if __name__ == "__main__":
    config = GESConfig(K=64, EXPLORATION_TIMES=1500)
    env = GridWorldEnv()

    print("Starting GE-S with all optimizations...")
    # 1. 运行主算法并接收返回的存档 Run the main algorithm and receive the returned archive
    final_archive, optimal_trajectory_archive = run_ges_with_fov(env, config)

    # 2. 从最终存档中找到最优节点 Find the optimal nodes from the final archive
    best_nodes = []
    max_overall_value = -1
    if optimal_trajectory_archive:
        for node in optimal_trajectory_archive.values():
            if node.value > max_overall_value:
                max_overall_value = node.value
                best_nodes = [node]
            elif node.value == max_overall_value:
                best_nodes.append(node)
    else:
        # 如果最优轨迹档案库为空，则回退到原始方法
        print("Optimal trajectory archive is empty, falling back to searching all archives...")
        for archive_id, archive_data in final_archive.items():
            for node in archive_data['cells_pool'].values():
                if node.value > max_overall_value:
                    max_overall_value = node.value
                    best_nodes = [node]
                elif node.value == max_overall_value:
                    best_nodes.append(node)



    # 3. 如果找到了，就提取信息并调用新的可视化函数 If found, extract information and call the new visualization function
    if best_nodes:
        print(
            f"\nAnalysis Complete: Found {len(best_nodes)} optimal trajectories with a maximum value of {max_overall_value}.")

        # 提取所有最优路径 Extract all optimal paths
        optimal_paths = [node.path_states for node in best_nodes]

        # 提取观测集合 (所有最优节点的观测集都一样大) Extract the observation set (the observation sets of all optimal nodes are the same size)
        final_observed_set = best_nodes[0].observed_set

        # 调用渲染函数 Call the rendering function
        render_optimal_trajectories(env, optimal_paths, final_observed_set)
    else:
        print("\nAnalysis Complete: No valid trajectories were found in the archive.")