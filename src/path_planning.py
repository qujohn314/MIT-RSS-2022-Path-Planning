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
        self.start = None
        self.end = None
        self.obstacles = []

        self.odom_topic = rospy.get_param("~odom_topic")
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)


    def map_cb(self, msg):
        data = msg.data
        grid = data.reshape((msg.info.height, msg.info.width))
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                if grid[y][x] != 0: # this number is randomly chosen
                    self.obstacles.append(self.convert_xy_to_uv(msg, x, y))
        self.map_acquired = True
        pass ## REMOVE AND FILL IN ##

    def convert_xy_to_uv(self, msg, x, y):
        scaled_x = float(x)/msg.info.resolution
        scaled_y = float(y)/msg.info.resolution
        coordinates = np.array([scaled_x, scaled_y, 1])
        Q = msg.info.origin.orientation
        q0 = Q.x
        q1 = Q.y
        q2 = Q.z
        q3 = Q.w
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
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                            
        rotated = np.matmul(rot_matrix, coordinates.T)
        translation = np.array([msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.position.z])
        return rotated + translation.T

    def odom_cb(self, msg):
        # pass ## REMOVE AND FILL IN ##
        if not self.map_acquired:
            return
        x = msg.twist.twist.linear.x
        y = msg.twist.twist.linear.y
        theta = msg.twist.twist.angular.z
        self.start = [x,y,theta]

    def goal_cb(self, msg):
        pass ## REMOVE AND FILL IN ##

    def plan_path(self, start_point, end_point, map):
        ## CODE FOR PATH PLANNING ##
        # Create start and end node
        start_node = Node(None, start_point)
        start_point.g = start_node.h = start_node.f = 0
        end_node = Node(None, end_point)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        # Loop until you find the end
        while len(open_list) > 0:

            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1] # Return reversed path

            # Generate children
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(map) - 1) or node_position[0] < 0 or node_position[1] > (len(map[len(map)-1]) -1) or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if map[node_position[0]][node_position[1]] != 0:
                    continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(child)

        # convert the path to a trajectory
        trajectory = [] #initialize series of piecewise points
        for i in range(len(path)-1):
            current_node = path[i]
            next_node = path[i+1]

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

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rospy.spin()
