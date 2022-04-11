#!/usr/bin/env python

import rospy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from queue import PriorityQueue

class PathPlan(object):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """
    def __init__(self):
        self.map_acquired = False
        self.rot_matrix = None
        self.translation = None
        self.start = None
        self.end = None
        self.grid = None

        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)


    def map_cb(self, msg):
        self.create_rot_matrix
        data = msg.data.reshape((msg.info.height, msg.info.width))
        pixel_grid = np.zeros((int(msg.info.height/float(msg.info.resolution)), int(msg.info.width/float(msg.info.resolution))))
        for u in range(len(pixel_grid)):
            for v in range(len(pixel_grid[u])):
                x, y = self.convert_uv_to_xy(msg, u, v)
                pixel_grid[v, u] = data[y, x]
        self.grid = pixel_grid
        self.map_acquired = True
        # pass ## REMOVE AND FILL IN ##

    def create_rot_matrix(self, msg):
        Q = msg.info.origin.orientation
        q0 = Q.x
        q1 = Q.y
        q2 = Q.z
        q3 = Q.w
        # First row of the rotation matrix
        r00 = 2 * (q0*q0 + q1*q1) - 1
        r01 = 2 * (q1*q2 - q0*q3)
        r02 = 2 * (q1*q3 + q0*q2)
        # Second row of the rotation matrix
        r10 = 2 * (q1*q2 + q0*q3)
        r11 = 2 * (q0*q0 + q2*q2) - 1
        r12 = 2 * (q2*q3 - q0*q1)
        # Third row of the rotation matrix
        r20 = 2 * (q1*q3 - q0*q2)
        r21 = 2 * (q2*q3 + q0*q1)
        r22 = 2 * (q0*q0 + q3*q3) - 1
        
        # 3x3 rotation matrix
        self.rot_matrix = np.array([[r00, r01, r02],
                                    [r10, r11, r12],
                                    [r20, r21, r22]])
        self.translation = np.array([msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.position.z])

    def convert_xy_to_uv(self, msg, x, y):
        # Divide (x,y) by map resolution
        scaled_x = float(x)/msg.info.resolution
        scaled_y = float(y)/msg.info.resolution
        # Get new set of (x,y coordinates)
        coordinates = np.array([scaled_x, scaled_y, 1])
        # First apply the rotation (rotation @ coordinates)
        rotated = np.matmul(self.rot_matrix, coordinates.T)
        # Then apply the translation 
        shifted = rotated + self.translation
        # Only return the (u,v) point because last point is unnecessary
        return shifted[:2]

    def convert_uv_to_xy(self, msg, x, y):
        # Divide (x,y) by map resolution
        scaled_x = x * msg.info.resolution
        scaled_y = y * msg.info.resolution
        # Get new set of (x,y coordinates)
        coordinates = np.array([scaled_x, scaled_y, 1])
        # First apply the rotation (rotation @ coordinates)
        rotated = np.matmul(self.rot_matrix, coordinates.T)
        # Then apply the translation 
        shifted = rotated + self.translation
        # Only return the (u,v) point because last point is unnecessary
        return shifted[:2]

    def odom_cb(self, msg):
        # pass ## REMOVE AND FILL IN ##
        if not self.map_acquired:
            return
        x = msg.twist.twist.linear.x
        y = msg.twist.twist.linear.y
        # theta = msg.twist.twist.angular.z
        self.start = np.array([x,y])

    def goal_cb(self, msg):
        # pass ## REMOVE AND FILL IN ##
        if not self.map_acquired:
            return
        x = msg.twist.twist.linear.x
        y = msg.twist.twist.linear.y
        # theta = msg.twist.twist.angular.z
        self.goal = np.array([x,y])

    def heuristic(self, a, b):
        return (b[0]-a[0])**2 + (b[1]-a[1])**2

    def generate_neighbors(self, node):
        north = np.array([0,1])
        east = np.array([1,0])
        south = np.array([0,-1])
        west = np.array([-1,0])
        return [node+north, node+east, node+south, node+west]

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##
        if not self.start or not self.end:
            return
        found_goal = False
        open_list = PriorityQueue()
        final_path = []
        came_from = {}
        cost_so_far = {}
        came_from[start_point] = None
        cost_so_far[start_point] = 0
        # Create start and end node

        open_list.put((0, start_point))

        # Loop until you find the end
        while not open_list.empty():
            current = open_list.get()[1]
            
            if (current==end_point).all():
                found_goal = True
                break
            
            neighbors = self.generate_neighbors(current)
            for neighbor in neighbors: # Adjacent squares
                if map[neighbor[0]][neighbor[1]] != 0:
                    continue
                if neighbor[0] > (len(map) - 1) or neighbor[0] < 0 or neighbor[1] > (len(map[0])-1) or neighbor[1] < 0:
                    continue
                
                new_cost = cost_so_far[current] + 1
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, end_point)
                    open_list.put(priority, neighbor)
                    came_from[neighbor] = current
            

        # Found the goal
        if found_goal:
            current = end_point
            path = [current]
            while current is not None:
                path.append(came_from[current])
                current = came_from[current]
            final_path = path[::-1] # Return reversed path

        # convert the path to a trajectory
        trajectory = [] #initialize series of piecewise points
        for i in range(len(final_path)-1):
            current_node = final_path[i]
            next_node = final_path[i+1]

            delta_x = next_node.position[0] - current_node.position[0]
            delta_y = next_node.position[1] - current_node.position[1]
            delta_theta = delta_theta_r = math.tan(delta_y/delta_x)

            #convert theta to robot coordinate frame
            rotation_matrix = np.array()
            delta_W = np.dot(np.array(delta_x, delta_y, delta_theta), rotation_matrix)
            
            trajectory.append(delta_W)


        # publish trajectory
        self.traj_pub.publish(self.trajectory.toPoseArray())

        # visualize trajectory Markers
        self.trajectory.publish_viz()

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
