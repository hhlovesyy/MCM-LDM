import numpy as np
import heapq
# 只要装了 scipy，这里绝不会报错
from scipy.interpolate import splprep, splev 

class PathPlanner:
    def __init__(self, world_range=20.0, grid_res=0.2, margin=0.5):
        """
        :param world_range: 世界范围 (米)
        :param grid_res: 网格分辨率 (米/格)
        :param margin: 避障安全距离 (米)
        """
        self.world_range = world_range
        self.res = grid_res
        self.margin = margin
        self.grid_size = int(world_range / grid_res)
        self.offset = world_range / 2.0
        
        # 初始化网格 (0:空闲, 1:障碍)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

    def _to_grid(self, x, y):
        """物理坐标 -> 网格坐标"""
        gx = int((x + self.offset) / self.res)
        gy = int((y + self.offset) / self.res)
        gx = max(0, min(self.grid_size - 1, gx))
        gy = max(0, min(self.grid_size - 1, gy))
        return gx, gy

    def _to_world(self, gx, gy):
        """网格坐标 -> 物理坐标"""
        wx = (gx * self.res) - self.offset
        wy = (gy * self.res) - self.offset
        return wx, wy

    def set_obstacles(self, obstacles):
        """
        映射障碍物
        obstacles: list of dict
          - Cylinder: {'type': 'cylinder', 'center': [x, z], 'radius': r}
          - Box:      {'type': 'box', 'center': [x, z], 'extent': [width, depth]}
        """
        self.grid.fill(0) 
        
        # 生成网格的物理坐标矩阵
        y_idxs, x_idxs = np.indices((self.grid_size, self.grid_size))
        world_xs = (x_idxs * self.res) - self.offset
        world_ys = (y_idxs * self.res) - self.offset
        
        for obs in obstacles:
            ox, oy = obs['center']
            
            if obs['type'] == 'cylinder':
                # 圆形逻辑
                r = obs['radius'] + self.margin 
                dist = np.sqrt((world_xs - ox)**2 + (world_ys - oy)**2)
                self.grid[dist <= r] = 1
                
            elif obs['type'] == 'box':
                # === 🔥 新增：矩形逻辑 🔥 ===
                # extent = [width, depth]
                w, d = obs['extent']
                # 考虑安全边距
                half_w = (w / 2.0) + self.margin
                half_d = (d / 2.0) + self.margin
                
                # 判断点是否在矩形范围内 (Axis-Aligned Bounding Box)
                mask_x = np.abs(world_xs - ox) <= half_w
                mask_y = np.abs(world_ys - oy) <= half_d
                self.grid[mask_x & mask_y] = 1

    def _astar(self, start, end):
        """标准 A* 算法"""
        start_node = self._to_grid(*start)
        end_node = self._to_grid(*end)
        
        # 检查起点终点是否在障碍物内
        if self.grid[start_node[1], start_node[0]] == 1:
            return [start]
        if self.grid[end_node[1], end_node[0]] == 1:
            return [end]

        pq = [(0, start_node[0], start_node[1])]
        visited = set()
        came_from = {}
        g_score = {start_node: 0}
        
        while pq:
            current_cost, cx, cy = heapq.heappop(pq)
            current = (cx, cy)
            
            if current == end_node:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_node)
                return path[::-1] # Reverse
            
            if current in visited:
                continue
            visited.add(current)
            
            # 8连通
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)
                
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[ny, nx] == 1: continue
                    move_cost = 1.414 if dx!=0 and dy!=0 else 1.0
                    new_g = g_score[current] + move_cost
                    
                    if neighbor not in g_score or new_g < g_score[neighbor]:
                        g_score[neighbor] = new_g
                        h = np.sqrt((nx - end_node[0])**2 + (ny - end_node[1])**2)
                        heapq.heappush(pq, (new_g + h, nx, ny))
                        came_from[neighbor] = current
        
        return [start_node, end_node]

    def generate_path(self, waypoints, obstacles):
        """生成最终平滑路径"""
        self.set_obstacles(obstacles)
        full_grid_path = []
        
        for i in range(len(waypoints) - 1):
            p_start = waypoints[i]
            p_end = waypoints[i+1]
            segment = self._astar(p_start, p_end)
            if i > 0: segment = segment[1:] 
            full_grid_path.extend(segment)
            
        if len(full_grid_path) < 2:
            return np.array(waypoints)

        raw_path = np.array([self._to_world(p[0], p[1]) for p in full_grid_path])
        
        # B-Spline 平滑
        if len(raw_path) > 3:
            try:
                tck, u = splprep(raw_path.T, u=None, s=0.5, k=3) 
                u_new = np.linspace(u.min(), u.max(), 200) 
                smooth_path = np.array(splev(u_new, tck)).T
                return smooth_path
            except Exception as e:
                print(f"Spline Error: {e}")
                return raw_path
        else:
            return raw_path