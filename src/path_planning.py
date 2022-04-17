#!/usr/bin/env python

import rospy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose, Quaternion, Point
from nav_msgs.msg import Odometry, OccupancyGrid
import rospkg
import time, os
from utils import LineTrajectory
from Queue import PriorityQueue
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker

from tf.transformations import quaternion_from_matrix
import tf.transformations
from scipy.spatial.transform import Rotation as R
from skimage.morphology import disk
from skimage.morphology import dilation
from skimage.morphology import erosion

class VisualizationTools:

    def plot_line(self, x, y, color, publisher, frame = "/map"):
        """
        Publishes the points (x, y) to publisher
        so they can be visualized in rviz as
        connected line segments.
        Args:
            x, y: The x and y values. These arrays
            must be of the same length.
            publisher: the publisher to publish to. The
            publisher must be of type Marker from the
            visualization_msgs.msg class.
            color: the RGB color of the plot.
            frame: the transformation frame to plot in.
        """
        # Construct a line
        line_strip = Marker()
        line_strip.type = line_strip.LINE_STRIP
        line_strip.header.frame_id = frame

        # Set the size and color
        line_strip.scale.x = 0.1
        line_strip.scale.y = 0.1
        line_strip.color.a = 1.
        line_strip.color.r = color[0]
        line_strip.color.g = color[1]
        line_strip.color.b = color[2]
        
        # Fill the line with the desired values
        for xi, yi in zip(x, y):
            p = Point()
            p.x = xi
            p.y = yi
            line_strip.points.append(p)

        # Publish the line
        publisher.publish(line_strip)

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
        self.x_points = []
        self.y_points = []

        self.odom_topic = rospy.get_param("~odom_topic")
        # self.odom_topic = "/odom"
        self.initial_pose = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initial_pose_cb)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)

        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_cb)
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.start_point = rospy.Publisher("/planned_trajectory/start_point", Marker, queue_size=10)
        self.end_point = rospy.Publisher("/planned_trajectory/end_pose", Marker, queue_size=10)
        self.path = rospy.Publisher("/planned_trajectory/path", Marker, queue_size=1)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory/current", PoseArray, queue_size=10)
        

        self.new_path_to_create = True


    def map_cb(self, msg):
        self.create_rot_matrix(msg)
        self.resolution = msg.info.resolution

        data = msg.data
        data = np.reshape(np.asarray(data), (msg.info.height, msg.info.width))
        # rospy.loginfo(data)
        data[data == -1] = 21
        data[data < 20] = 0
        data[data > 20] = 1

        # idx = np.where(0 <= data < 20)
        # data = data[idx]

        footprint = disk(7)
        new_image = dilation(data, footprint) #erode map

        self.grid = new_image
        self.map_acquired = True

    def create_rot_matrix(self, msg):
        Q = msg.info.origin.orientation
        q0 = Q.x
        q1 = Q.y
        q2 = Q.z
        q3 = Q.w

        r = R.from_quat([q0, q1, q2, q3])
        self.rot_matrix = r.as_dcm()
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

        return (np.rint(u).astype(np.int32),np.rint(v).astype(np.int32))

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
        return (shifted[0], shifted[1])
    
    def euler_to_quat(self, euler,deg=False):
        r = R.from_euler('xyz', euler,degrees=deg)
        return r.as_quat()

    def odom_cb(self, msg):
        # pass ## REMOVE AND FILL IN ##
        if not self.map_acquired or (self.start is not None and self.end is not None):
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        # theta = msg.twist.twist.angular.z
        self.start = self.convert_xy_to_uv((x,y))
        self.x_points = []
        self.y_points = []

        start_point_marker = Marker()
        start_point_marker.header.frame_id = '/map'
        start_point_marker.type = start_point_marker.SPHERE
        start_point_marker.action = start_point_marker.ADD
        start_point_marker.scale.x = 0.4
        start_point_marker.scale.y = 0.4
        start_point_marker.scale.z = 0.4
        
        start_point_marker.pose.position.x = x
        start_point_marker.pose.position.y = y
        start_point_marker.pose.position.z = 0

        start_point_marker.pose.orientation = Quaternion(0,0,0,1)

        start_point_marker.color.a = 0.5
        start_point_marker.color.r = 0.0
        start_point_marker.color.g = 1.0
        start_point_marker.color.b = 0.0


        self.new_path_to_create = True
        self.start_point.publish(start_point_marker)
    
    def initial_pose_cb(self, msg):
        if not self.map_acquired:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.start = self.convert_xy_to_uv((x,y))
        self.x_points = []
        self.y_points = []

        start_point_marker = Marker()
        start_point_marker.header.frame_id = '/map'
        start_point_marker.type = start_point_marker.SPHERE
        start_point_marker.action = start_point_marker.ADD
        start_point_marker.scale.x = 0.4
        start_point_marker.scale.y = 0.4
        start_point_marker.scale.z = 0.4
        
        start_point_marker.pose.position.x = x
        start_point_marker.pose.position.y = y
        start_point_marker.pose.position.z = 0

        start_point_marker.pose.orientation = Quaternion(0,0,0,1)

        start_point_marker.color.a = 0.5
        start_point_marker.color.r = 0.0
        start_point_marker.color.g = 1.0
        start_point_marker.color.b = 0.0


        self.new_path_to_create = True
        self.start_point.publish(start_point_marker)

    def goal_cb(self, msg):
        # pass ## REMOVE AND FILL IN ##
        if not self.map_acquired:
            return
        x = msg.pose.position.x
        y = msg.pose.position.y
        # theta = msg.twist.twist.angular.z
        if self.end is not None and self.start is not None:
            self.start = self.end
        self.end = self.convert_xy_to_uv((x,y))

        end_point_marker = Marker()
        end_point_marker.header.frame_id = '/map'
        end_point_marker.type = end_point_marker.SPHERE
        end_point_marker.action = end_point_marker.ADD
        end_point_marker.scale.x = 0.4
        end_point_marker.scale.y = 0.4
        end_point_marker.scale.z = 0.4

        
        
        end_point_marker.pose.position.x = x
        end_point_marker.pose.position.y = y
        end_point_marker.pose.position.z = 0

        end_point_marker.pose.orientation = Quaternion(0,0,0,1)
        end_point_marker.color.a = 0.5
        end_point_marker.color.r = 1.0
        end_point_marker.color.g = 0
        end_point_marker.color.b = 0.0

        self.new_path_to_create = True
        self.end_point.publish(end_point_marker)


    def heuristic(self, a, b):
        return ((b[0]-a[0])**2 + (b[1]-a[1])**2)**(0.5)
        # '''
        # dx = abs(a[0] - b[0])
        # dy = abs(a[1] - b[1])
        # D2 = 2**0.5
        #
        # h = (dx+dy) + (D2 - 2)*min(dx,dy)
        # return h '''

    def generate_neighbors(self, node):
        north = (0, 1)
        northeast = (1, 1)
        east = (1, 0)
        southeast = (1, -1)
        south = (0, -1)
        southwest = (-1,-1)
        west = (-1,0)
        northwest = (-1,1)

        norther = (-1,2)
        northeaster = (1,2)
        easter = (2,1)
        southeaster = (2,-1)
        souther = (1,-2)
        southwester = (-1,-2)
        wester = (-2,-1)
        northwester = (-2,1)
        directions = [north, northeast, east, southeast, south, southwest, west, northwest,norther, northeaster, easter, southeaster, souther, southwester, wester, northwester ]
        
        neighbors = []
        for direction in directions:
            neighbors.append((node[0]+direction[0], node[1]+direction[1]))
        return neighbors

    def plan_path(self, start_point, end_point, grid):
        ## CODE FOR PATH PLANNING ##

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
            
            if current==end_point:
                found_goal = True
                break
            
            neighbors = self.generate_neighbors(current)
            for neighbor in neighbors: # Adjacent squares
                
                # out of bounds
                if neighbor[0] > (len(grid[0]) - 1) or neighbor[0] < 0 or neighbor[1] > (len(grid)-1) or neighbor[1] < 0:
                    # rospy.loginfo("out of bounds detected :(")
                    # rospy.loginfo("uv coord:" + str((neighbor[0],neighbor[1])))
                    continue

                # obstacle in the way
                if grid[neighbor[1]][neighbor[0]] > 0 or grid[neighbor[1]][neighbor[0]] == -1:
                    continue
                
                # Increase the cost by one
                new_cost = cost_so_far[current] + ((current[0]-neighbor[0])**2 + (current[1]-neighbor[1])**2)**(0.5)
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, end_point)
                    open_list.put((priority, neighbor))
                    came_from[neighbor] = current

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
        vector_arr = []
        map_x = []
        map_y = []
        poseArray = PoseArray()
        poseArray.header.frame_id = "/map"
        for i in range(1, len(final_path)-1):
            current_node = final_path[i]
            next_node = final_path[i+1]

            # Publish the point
            x, y = self.convert_uv_to_xy((current_node[0],current_node[1]))
            map_x.append(x)
            map_y.append(y)
            next_x, next_y = self.convert_uv_to_xy((next_node[0],next_node[1]))

            delta_x = next_x - x
            delta_y = next_y - y
            vector1 = (delta_x, delta_y)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0

            vector_arr.append(vector1)

            rotation = math.atan2(delta_y,delta_x)

            quat_array = self.euler_to_quat([0,0,rotation])
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


        visualize = VisualizationTools()
        self.x_points = self.x_points + map_x 
        self.y_points = self.y_points + map_y
        rospy.loginfo("points")
        visualize.plot_line(self.x_points, self.y_points, [0,1,0], self.path, frame='/map')

if __name__=="__main__":
    rospy.init_node("path_planning")
    pf = PathPlan()
    rate = rospy.Rate(50)
    while pf.start is None or pf.end is None:
        continue

    while not rospy.is_shutdown():
        if(pf.new_path_to_create):
            pf.plan_path(pf.start, pf.end, pf.grid)
            pf.new_path_to_create = False
        rate.sleep()
    
