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
        # Distance de sécurité en cellules (2 cellules = 10cm avec résolution 0.05)
        self.inflation_radius = 6

    def set_map(self, occupancy_grid):
        """Configure la carte et ses paramètres"""
        self.map_data = np.array(occupancy_grid.data).reshape(
            (occupancy_grid.info.height, occupancy_grid.info.width))
        self.resolution = occupancy_grid.info.resolution
        self.origin = occupancy_grid.info.origin
        self.width = occupancy_grid.info.width
        self.height = occupancy_grid.info.height
        
        # Inflation des obstacles
        self.inflate_obstacles()

    def inflate_obstacles(self):
        """Ajoute une marge de sécurité autour des obstacles"""
        inflated_map = np.copy(self.map_data)
        for y in range(self.height):
            for x in range(self.width):
                if self.map_data[y, x] > 50:  # Si c'est un obstacle
                    for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                        for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                            if (0 <= y + dy < self.height and 
                                0 <= x + dx < self.width):
                                inflated_map[y + dy, x + dx] = 100
        self.map_data = inflated_map

    def world_to_map(self, x, y):
        """Convertit les coordonnées monde en coordonnées grille"""
        mx = int((x - self.origin.position.x) / self.resolution)
        my = int((y - self.origin.position.y) / self.resolution)
        return mx, my

    def map_to_world(self, mx, my):
        """Convertit les coordonnées grille en coordonnées monde"""
        x = mx * self.resolution + self.origin.position.x
        y = my * self.resolution + self.origin.position.y
        return x, y

    def get_neighbors(self, current):
        """Retourne les cellules voisines valides"""
        x, y = current
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), 
                       (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                self.map_data[ny, nx] < 50):
                neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, a, b):
        """Distance euclidienne comme heuristique"""
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def find_path(self, start, goal):
        """Implémentation de l'algorithme A*"""
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

        # Reconstruction du chemin
        path = []
        current = goal_map
        if goal_map in came_from:
            while current is not None:
                x, y = self.map_to_world(current[0], current[1])
                path.append((x, y))
                current = came_from[current]
            path.reverse()
            return path
        return None