
import os
import imageio
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import matplotlib.patches as patches
import env
import plotting
import utils


# Initialize a node with the given coordinates
class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None

# Initialize the IRrtStar class for path planning
class IRrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max):
        self.x_start = Node(x_start)  #Start node
        self.x_goal = Node(x_goal) #Goal node
        self.step_len = step_len # Maximum step size
        self.goal_sample_rate = goal_sample_rate # Probability to directly sample the goal
        self.search_radius = search_radius # Maximum search radius for neighbors
        self.iter_max = iter_max # Maximum iterations for the planning loop

        self.environment = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.fig, self.ax = plt.subplots()
        self.delta = self.utils.delta
        self.x_range = self.environment.x_range
        self.y_range = self.environment.y_range
        self.obs_circle = self.environment.obs_circle
        # self.obs_rectangle = self.environment.obs_rectangle
        self.obs_boundary = self.environment.obs_boundary

        # Initialize node and path-related data
        self.V = [self.x_start]
        self.X_soln = set()
        self.path = None

    def init(self):
        cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
                            [(self.x_start.y + self.x_goal.y) / 2.0], [0.0]])
        x_best = self.x_start

        return theta, cMin, xCenter, C, x_best
    #Bezier curve implementation using control points
    def bezier_curve(self, P0, P1, P2, P3, n_points=200):
        t = np.linspace(0, 1, n_points).reshape(1, -1)  # Shape (1, n_points)
    
        P0 = P0.reshape(-1, 1)  # Shape (2, 1)
        P1 = P1.reshape(-1, 1)  # Shape (2, 1)
        P2 = P2.reshape(-1, 1)  # Shape (2, 1)
        P3 = P3.reshape(-1, 1)  # Shape (2, 1)

        curve = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
        return curve.T 
    
    def get_control_points(self, P0, P3):
        curvature_factor = 0.11
        distance = np.linalg.norm(np.array(P3) - np.array(P0))
        curvature = min(curvature_factor * distance, 1)  # Limit maximum curvature
        # Find the midpoint between P0 and P3
        mid_x = (P0[0] + P3[0]) / 2
        mid_y = (P0[1] + P3[1]) / 2

        # Offset control points to introduce curvature
        offset_x = curvature_factor * (P3[1] - P0[1])
        offset_y = curvature_factor * (P0[0] - P3[0])

        P1 = (mid_x + offset_x, mid_y + offset_y)
        P2 = (mid_x - offset_x, mid_y - offset_y)

        return np.array(P1), np.array(P2)

     # Main function to plan the path using Informed RRT* algorithm
    def planning(self):
        theta, dist, x_center, C, x_best = self.init()
        c_best = np.inf

        # Run planning loop for the specified number of iterations
        for k in range(self.iter_max):
            # If a solution exists, find the best solution node
            if self.X_soln:
                cost = {node: self.Cost(node) for node in self.X_soln}
                x_best = min(cost, key=cost.get)
                c_best = cost[x_best]

            x_rand = self.Sample(c_best, dist, x_center, C)   # Generate a random sample node within the informed ellipse or the free space
            x_nearest = self.Nearest(self.V, x_rand) # Find the nearest node to the random node in the tree
            x_new = self.Steer(x_nearest, x_rand)  # Create a new node moving towards the random node from the nearest node
            # If the new node is valid and collision-free, add it to the tree
            if x_new and not self.utils.is_collision(x_nearest, x_new):
                X_near = self.Near(self.V, x_new)
                c_min = self.Cost(x_nearest) + self.Line(x_nearest, x_new)
                self.V.append(x_new)

                # choose parent
                for x_near in X_near:
                    c_new = self.Cost(x_near) + self.Line(x_near, x_new)
                    if c_new < c_min:
                        x_new.parent = x_near
                        c_min = c_new

                 # Rewire the neighbors if connecting through `x_new` offers a cheaper path
                for x_near in X_near:
                    c_near = self.Cost(x_near)
                    c_new = self.Cost(x_new) + self.Line(x_new, x_near)
                    if c_new < c_near:
                        x_near.parent = x_new
                # If the new node is within the goal region, add it to the solutions
                if self.InGoalRegion(x_new):
                    if not self.utils.is_collision(x_new, self.x_goal):
                        self.X_soln.add(x_new)
            # Periodically update the animation with current planning progress
            if k % 20 == 0:
                self.animation(x_center=x_center, c_best=c_best,
                               dist=dist, theta=theta)
        # Extract the best path found and convert to a Bezier curve
        self.path = self.ExtractPath(x_best)
        bezier_path = []
        for i in range(0, len(self.path) - 1, 1):
            P0 = np.array(self.path[i])
            P3 = np.array(self.path[i+1])
            P1, P2 = self.get_control_points(P0, P3)

            bezier_path.extend(self.bezier_curve(P0, P1, P2, P3))

        plt.plot([p[0] for p in bezier_path], [p[1] for p in bezier_path], '-r')
        plt.pause(0.01)
        plt.show()

         # Print the total path cost
        total_cost = self.Cost(x_best)
        plt.text(0.02, 0.02, f'Total Path Cost: {total_cost:.2f}', fontsize=12, color='blue', transform=plt.gca().transAxes)
        print(f"Total path cost: {total_cost}")

    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)   # Create a new node in the direction from `x_start` to `x_goal`
        dist = min(self.step_len, dist)
        # Create a new node with the calculated coordinates
        node_new = Node((x_start.x + dist * math.cos(theta),
                         x_start.y + dist * math.sin(theta)))
        node_new.parent = x_start

        return node_new

    def Near(self, nodelist, node):
        n = len(nodelist) + 1
        r = 50 * math.sqrt((math.log(n) / n))

        dist_table = [(nd.x - node.x) ** 2 + (nd.y - node.y)
                      ** 2 for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if
                  dist_table[ind] <= r ** 2
                  and not self.utils.is_collision(nodelist[ind], node)]

        return X_near

    def Sample(self, c_max, c_min, x_center, C):
        if c_max < np.inf:
            d = c_max ** 2 - c_min ** 2
            if d < 0:
                d = 0.01
            r = [c_max / 2.0,
                 math.sqrt(d) / 2.0,
                 math.sqrt(d) / 2.0]
            L = np.diag(r)

            while True:
                x_ball = self.SampleUnitBall()
                x_rand = np.dot(np.dot(C, L), x_ball) + x_center
                if self.x_range[0] + self.delta <= x_rand[0] <= \
                        self.x_range[1] - self.delta and self.y_range[0] + \
                        self.delta <= x_rand[1] <= self.y_range[1]-self.delta:
                    break
            x_rand = Node((x_rand[(0, 0)], x_rand[(1, 0)]))
        else:
            x_rand = self.SampleFreeSpace()

        return x_rand

    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])

    def SampleFreeSpace(self):
        delta = self.delta

        if np.random.random() > self.goal_sample_rate:
            return Node((np.random.uniform
                         (self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta,
                                           self.y_range[1] - delta)))

        return self.x_goal

    def ExtractPath(self, node):
        path = [[self.x_goal.x, self.x_goal.y]]

        while node.parent:
            path.append([node.x, node.y])
            node = node.parent

        path.append([self.x_start.x, self.x_start.y])

        return path

    def InGoalRegion(self, node):
        if self.Line(node, self.x_goal) < self.step_len:
            return True

        return False

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array([[(x_goal.x - x_start.x) / L],
                       [(x_goal.y - x_start.y) / L], [0.0]])
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U)
                        * np.linalg.det(V_T.T)]) @ V_T

        return C

    @staticmethod
    def Nearest(nodelist, n):
        return nodelist[int(np.argmin([(nd.x - n.x) ** 2 + (nd.y - n.y) ** 2
                                       for nd in nodelist]))]

    @staticmethod
    def Line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    def Cost(self, node):
        if node == self.x_start:
            return 0.0

        if node.parent is None:
            return np.inf

        cost = 0.0
        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
    
    def generate_sinusoidal_bezier_points(P0, P3, amplitude=0.2, frequency=1.0):
        # Midpoint between the two nodes
        mid_x = (P0[0] + P3[0]) / 2
        mid_y = (P0[1] + P3[1]) / 2
        
        # Direction vector and orthogonal vector for sinusoidal transformation
        direction = np.array([P3[0] - P0[0], P3[1] - P0[1]])
        direction_length = np.linalg.norm(direction)
        orthogonal = np.array([-direction[1], direction[0]]) / direction_length

        # Apply sinusoidal transformation to control points
        offset = amplitude * np.sin(frequency * np.pi)
        P1 = np.array([mid_x + orthogonal[0] * offset, mid_y + orthogonal[1] * offset])
        P2 = np.array([mid_x - orthogonal[0] * offset, mid_y - orthogonal[1] * offset])
        return np.array(P0), P1, P2, np.array(P3)

    def animation(self, x_center=None, c_best=None, dist=None, theta=None):
        # Update and draw the animation for the current state of planning
        plt.cla()
        self.plot_grid("Informed RRT* - Bezier Spline")
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
         # Draw all tree edges as Bezier curves
        for node in self.V:
            if node.parent:
                P0 = np.array([node.parent.x, node.parent.y])
                P3 = np.array([node.x, node.y])
                P1, P2 = self.get_control_points(P0, P3)
            
                bezier_curve = self.bezier_curve(P0, P1, P2, P3)
            
                plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], "-", color="#90EE90")
        # Draw the informed ellipse indicating the search space
        if c_best != np.inf:
            self.draw_ellipse(x_center, c_best, dist, theta)

        plt.pause(0.01)
        frame_filename = f"frame_{len(os.listdir('frames'))}.png"
        plt.savefig(f"frames/{frame_filename}")

    def plot_grid(self, name):

        for (ox, oy, w, h) in self.obs_boundary:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='orange',
                    fill=True
                )
            )

        plt.plot(self.x_start.x, self.x_start.y, "ys", linewidth=3)
        plt.plot(self.x_goal.x, self.x_goal.y, "ks", linewidth=3)

        plt.title(name)
        plt.axis("equal")

    @staticmethod
    def draw_ellipse(x_center, c_best, dist, theta):
        temp = c_best ** 2 - dist ** 2
        if temp < 0:
            temp = 0.01
        a = math.sqrt(temp) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, ".b")
        plt.plot(px, py, linestyle='--', color='darkorange', linewidth=2)


if __name__ == '__main__':
    x_start = (5, 5)  # Starting node
    x_goal = (90, 90)  # Goal node

    rrt_star = IRrtStar(x_start, x_goal, 1, 0.10, 10, 1000)
    path = rrt_star.planning()
    print('Path Found:')
    print(path)

    frames = sorted([f"frames/{frame}" for frame in os.listdir("frames") if frame.endswith(".png")])
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave('planning_animation.gif', images, fps=10)

    # Clean up the frames
    for frame in frames:
        os.remove(frame)