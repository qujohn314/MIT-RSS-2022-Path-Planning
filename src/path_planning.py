#!/usr/bin/env python

import rospy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose, Quaternion
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from Queue import PriorityQueue
from scipy.spatial.transform import Rotation as R

from tf.transformations import quaternion_from_matrix
import tf.transformations
from scipy.spatial.transform import Rotation as R

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
        self.resolution = None

        self.odom_topic = rospy.get_param("~odom_topic")
        self.initial_pose = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initial_pose_cb)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)


    def map_cb(self, msg):
        self.create_rot_matrix(msg)
        self.resolution = msg.info.resolution
        # rospy.loginfo("data size")
        # rospy.loginfo(len(msg.data))
        # rospy.loginfo("height")
        # rospy.loginfo(msg.info.height)
        # rospy.loginfo("width")
        # rospy.loginfo(msg.info.width)
        # rospy.loginfo("resolution")
        # rospy.loginfo(msg.info.resolution)
        # rospy.loginfo("translation")
        # rospy.loginfo(self.translation)
        data = np.reshape(msg.data, (msg.info.height, msg.info.width))
        # pixel_grid = np.zeros((int(msg.info.height), int(msg.info.width)))
        # for u in range(len(pixel_grid)):
        #     for v in range(len(pixel_grid[u])):
        #         pixel_grid[v, u] = data[v, u]
        # self.grid = pixel_grid
        self.grid = data
        self.map_acquired = True

    def create_rot_matrix(self, msg):
        Q = msg.info.origin.orientation
        q0 = Q.x
        q1 = Q.y
        q2 = Q.z
        q3 = Q.w

        r = R.from_quat([q0, q1, q2, q3])
        self.rot_matrix = r.as_dcm()
        # # First row of the rotation matrix
        # r00 = 2 * (q0*q0 + q1*q1) - 1
        # r01 = 2 * (q1*q2 - q0*q3)
        # r02 = 2 * (q1*q3 + q0*q2)
        # # Second row of the rotation matrix
        # r10 = 2 * (q1*q2 + q0*q3)
        # r11 = 2 * (q0*q0 + q2*q2) - 1
        # r12 = 2 * (q2*q3 - q0*q1)
        # # Third row of the rotation matrix
        # r20 = 2 * (q1*q3 - q0*q2)
        # r21 = 2 * (q2*q3 + q0*q1)
        # r22 = 2 * (q0*q0 + q3*q3) - 1
        
        # # 3x3 rotation matrix
        # self.rot_matrix = np.array([[r00, r01, r02],
        #                             [r10, r11, r12],
        #                             [r20, r21, r22]])
        self.translation = np.array([msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.position.z])
        self.map_acquired = True
    def convert_xy_to_uv(self, x_y_coord):
        x = x_y_coord[0]
        y = x_y_coord[1]
        coordinates = np.array([x,y,0])
        #Apply translation
        shifted = coordinates - self.translation
        #Undo rotation
        rotated = np.matmul(self.rot_matrix.T, shifted.T).T #inverse rotation

        #convert to pixel coords
        u = float(rotated[0])/self.resolution
        v = float(rotated[1])/self.resolution

        return (np.rint(u).astype(np.uint16),np.rint(v).astype(np.uint16))

    def convert_uv_to_xy(self, u_v_coord):
        u = u_v_coord[0]
        v = u_v_coord[1]
        # Multiply (u,v) by map resolution
        scaled_u = u * self.resolution
        scaled_v = v * self.resolution
        # Get new set of (u,v coordinates)
        coordinates = np.array([scaled_u, scaled_v, 0])
        # First apply the rotation (rotation @ coordinates) 
        rotated = np.matmul(self.rot_matrix, coordinates.T).T #3x3 * 3x1
        # Then apply the translation 
        shifted = rotated + self.translation
        # Only return the (x,y) point because last point is unnecessary
        return (np.rint(shifted[0]).astype(np.uint16), np.rint(shifted[1]).astype(np.uint16))
    
    def euler_to_quat(self, euler,deg=False):
        r = R.from_euler('xyz', euler,degrees=deg)
        return r.as_quat()

    def odom_cb(self, msg):
        # pass ## REMOVE AND FILL IN ##
        if not self.map_acquired:
            return
        x = msg.twist.twist.linear.x
        y = msg.twist.twist.linear.y
        # theta = msg.twist.twist.angular.z
        self.start = (x,y)
    
    def initial_pose_cb(self, msg):
        if not self.map_acquired:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.start = (x,y)

    def goal_cb(self, msg):
        # pass ## REMOVE AND FILL IN ##
        if not self.map_acquired:
            return
        x = msg.pose.position.x
        y = msg.pose.position.y
        # theta = msg.twist.twist.angular.z
        self.end = (x,y)

    def heuristic(self, a, b):
        return (b[0]-a[0])**2 + (b[1]-a[1])**2

    def generate_neighbors(self, node):
        north = (0,1)
        northeast = (1,1)
        east = (1,0)
        southeast = (1,-1)
        south = (0,-1)
        southwest = (-1,-1)
        west = (-1,0)
        northwest = (-1,1)
        directions = [north, northeast, east, southeast, south, southwest, west, northwest]
        
        neighbors = []
        for direction in directions:
            neighbors.append((node[0]+direction[0], node[1]+direction[1]))
        return neighbors

    def plan_path(self, start_point, end_point, grid):
        ## CODE FOR PATH PLANNING ##
        start_point = self.convert_xy_to_uv(start_point)
        end_point = self.convert_xy_to_uv(end_point)

        found_goal = False
        open_list = PriorityQueue()
        final_path = []
        came_from = {}
        cost_so_far = {}
        came_from[start_point] = None
        cost_so_far[start_point] = 0
        # Create start and end node

        open_list.put((0, start_point))
        # rospy.loginfo(start_point)
        # rospy.loginfo(grid[start_point[1]][start_point[0]])
        # Loop until you find the end
        while not open_list.empty():
            current = open_list.get()[1]
            
            if current==end_point:
                found_goal = True
                break
            
            neighbors = self.generate_neighbors(current)
            for neighbor in neighbors: # Adjacent squares
                # obstacle in the way
                if grid[neighbor[1]][neighbor[0]] != 0:
                    continue
                # out of bounds
                if neighbor[0] > (len(grid) - 1) or neighbor[0] < 0 or neighbor[1] > (len(grid[0])-1) or neighbor[1] < 0:
                    continue
                
                # Increase the cost by one
                new_cost = cost_so_far[current] + 1
                # rospy.loginfo("list")
                # rospy.loginfo(cost_so_far)
                # rospy.loginfo("new_cost")
                # rospy.loginfo(new_cost)
                # rospy.loginfo("cost_so_far")
                # rospy.loginfo(cost_so_far[neighbor])
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, end_point)
                    open_list.put((priority, neighbor))
                    came_from[neighbor] = current
            # rospy.loginfo("Queue size")
            # rospy.loginfo(open_list.qsize())

        # Found the goal
        if found_goal:
            current = end_point
            path = [current]
            while current is not None:
                path.append(came_from[current])
                current = came_from[current]
            final_path = path[::-1] # Return reversed path
            # rospy.loginfo(final_path)
        else:
            rospy.loginfo("goal not found")

        # convert the path to a trajectory
        trajectory = [] #initialize series of piecewise points
        poseArray = PoseArray()
        for i in range(1, len(final_path)-1):
            current_node = final_path[i]
            next_node = final_path[i+1]

            x, y = self.convert_uv_to_xy((current_node[0],current_node[1]))
            next_x, next_y = self.convert_uv_to_xy((next_node[0],next_node[1]))

            delta_x = next_x - x
            delta_y = next_y - y
            delta_theta = math.atan2(next_y,next_x) - math.atan2(y,x)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0

            rotation = 0
            if delta_y > 0:
                rotation = 90
            elif delta_y < 0:
                rotation = 270

            quat_array = self.euler_to_quat([0,0,rotation],True)
            pose.orientation = Quaternion(quat_array[0],quat_array[1],quat_array[2],quat_array[3])


            #convert theta to robot coordinate frame
            #rotation_matrix = np.array()
            #delta_W = np.dot(np.array(delta_x, delta_y, delta_theta), rotation_matrix)
            
            trajectory.append(pose)


        poseArray.poses = trajectory
        # publish trajectory
        self.traj_pub.publish(poseArray)

        # visualize trajectory Markers
        self.trajectory.publish_viz()

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    while pf.start is None or pf.end is None:
        continue
    # rospy.loginfo("INITIALIZED AND RUNNING!!!")
    pf.plan_path(pf.start, pf.end, pf.grid)
    # rospy.loginfo("Path complete")
    rospy.spin()
