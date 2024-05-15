import math
import numpy as np
import os
import sys
import env
from rrt import Node


class Utils:
    def __init__(self):
        self.environment = env.Env()
        # Distance tolerance for intersection calculations
        self.delta = 0.5
        self.obs_circle = self.environment.obs_circle
        self.obs_boundary = self.environment.obs_boundary   # Retrieve boundary obstacles from the environment

     # Update the obstacle information with new data
    def update_obs(self, obs_cir, obs_bound, obs_rec):
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound

    # Placeholder function to return vertices of obstacles
    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]] # Vector from `a` to the origin `o`
        v2 = [b[0] - a[0], b[1] - a[1]]  # Vector from `a` to `b` (one side of the rectangle)
        v3 = [-d[1], d[0]] # Perpendicular vector to the line direction

        # Calculate the determinant to check if the line is parallel
        div = np.dot(v2, v3)

        # If determinant is zero, lines are parallel, no intersection
        if div == 0:
            return False

        # Calculate scalar parameters for intersection
        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        # Check if the intersection point lies within the segment and rectangle side
        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False
    
    # Check if a line segment intersects a circular obstacle
    def is_intersect_circle(self, o, d, a, r):
        d2 = np.dot(d, d)
        delta = self.delta

        if d2 == 0:
            return False

        t = np.dot([a[0] - o[0], a[1] - o[1]], d) / d2

        if 0 <= t <= 1:
            shot = Node((o[0] + t * d[0], o[1] + t * d[1]))
            if self.get_dist(shot, Node(a)) <= r + delta:
                return True

        return False
     # Check if either endpoint is inside an obstacle
    def is_collision(self, start, end):
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True

        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        for (x, y, r) in self.obs_circle:
            if self.is_intersect_circle(o, d, [x, y], r):
                return True

        return False

    # Check if the node is inside any defined obstacle
    def is_inside_obs(self, node):
        delta = self.delta

        for (x, y, r) in self.obs_circle:
            if math.hypot(node.x - x, node.y - y) <= r + delta:
                return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        return False

    # Get the origin and direction vector for a line segment
    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    # Calculate the Euclidean distance between two nodes
    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)
