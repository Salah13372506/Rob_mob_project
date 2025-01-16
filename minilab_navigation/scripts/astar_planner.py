#!/usr/bin/env python3

import numpy as np
from queue import PriorityQueue 
import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
import tf2_ros
import tf2_geometry_msgs

class AStarPlanner:
   def __init__(self):
       self.map_data = None
       self.resolution = None 
       self.origin = None
       self.width = None
       self.height = None
       self.inflation_radius = 7
       self.heuristic_type = "octile"  # Options: "manhattan", "diagonal", "octile", "chebyshev", "euclidean"

   def set_map(self, occupancy_grid):
       self.map_data = np.array(occupancy_grid.data).reshape(
           (occupancy_grid.info.height, occupancy_grid.info.width))
       self.resolution = occupancy_grid.info.resolution
       self.origin = occupancy_grid.info.origin
       self.width = occupancy_grid.info.width 
       self.height = occupancy_grid.info.height
       self.inflate_obstacles()

   def inflate_obstacles(self):
       inflated_map = np.copy(self.map_data)
       for y in range(self.height):
           for x in range(self.width):
               if self.map_data[y, x] > 50:
                   for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                       for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                           if (0 <= y + dy < self.height and 0 <= x + dx < self.width):
                               inflated_map[y + dy, x + dx] = 100
       self.map_data = inflated_map

   def world_to_map(self, x, y):
       mx = int((x - self.origin.position.x) / self.resolution)
       my = int((y - self.origin.position.y) / self.resolution) 
       return mx, my

   def map_to_world(self, mx, my):
       x = mx * self.resolution + self.origin.position.x
       y = my * self.resolution + self.origin.position.y
       return x, y

   def get_neighbors(self, current):
       x, y = current
       neighbors = []
       for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), 
                      (1, 1), (-1, 1), (1, -1), (-1, -1)]:
           nx, ny = x + dx, y + dy
           if (0 <= nx < self.width and 0 <= ny < self.height and 
               self.map_data[ny, nx] < 50):
               neighbors.append((nx, ny))
       return neighbors

   def smooth_path(self, path, window_size=5):
       if not path or len(path) < window_size:
           return path
           
       smoothed = []
       for i in range(len(path)):
           start_idx = max(0, i - window_size//2)
           end_idx = min(len(path), i + window_size//2 + 1)
           window = path[start_idx:end_idx]
           
           x_avg = sum(p[0] for p in window) / len(window)
           y_avg = sum(p[1] for p in window) / len(window)
           smoothed.append((x_avg, y_avg))
       
       return smoothed

   def heuristic(self, a, b):
        """
        Calcule l'heuristique entre deux points selon la méthode choisie
        a, b: tuples (x, y) représentant les coordonnées des points
        """
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        
        if self.heuristic_type == "manhattan":
            # Distance de Manhattan: plus rapide, surestime parfois
            return dx + dy
            
        elif self.heuristic_type == "diagonal":
            # Distance diagonale: bon compromis vitesse/précision
            return dx + dy + (np.sqrt(2) - 2) * min(dx, dy)
            
        elif self.heuristic_type == "octile":
            # Distance octile: similaire à diagonale mais plus rapide
            # Utilise 0.414 comme approximation de (√2 - 1)
            return dx + dy - 0.414 * min(dx, dy)
            
        elif self.heuristic_type == "chebyshev":
            # Distance de Chebyshev: très rapide, mais moins précise
            return max(dx, dy)
            
        else:  # "euclidean" - l'original
            # Distance euclidienne: plus précise mais plus lente
            return np.sqrt(dx * dx + dy * dy)
        

   def find_path(self, start, goal):
       start_map = self.world_to_map(start[0], start[1])
       goal_map = self.world_to_map(goal[0], goal[1])

       frontier = PriorityQueue()
       frontier.put((0, start_map))
       came_from = {start_map: None}
       cost_so_far = {start_map: 0}

       while not frontier.empty():
           current = frontier.get()[1]

           if current == goal_map:
               break

           for next in self.get_neighbors(current):
               new_cost = cost_so_far[current] + 1
               if next not in cost_so_far or new_cost < cost_so_far[next]:
                   cost_so_far[next] = new_cost
                   priority = new_cost + self.heuristic(goal_map, next)
                   frontier.put((priority, next))
                   came_from[next] = current

       path = []
       current = goal_map
       if goal_map in came_from:
           while current is not None:
               x, y = self.map_to_world(current[0], current[1])
               path.append((x, y))
               current = came_from[current]
           path.reverse()
           return self.smooth_path(path)
       return None
