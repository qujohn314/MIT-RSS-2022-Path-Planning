#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils
import tf

from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = rospy.get_param("~odom_topic")
        self.lookahead        = # FILL IN #
        self.speed            = # FILL IN #
        self.wrap             = # FILL IN #
        self.wheelbase_length = 0.55
        self.trajectory  = utils.LineTrajectory("/followed_trajectory")
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)
        self.drive_cmd = AckermannDriveStamped()
        self.drive_cmd.drive.speed = 1 #TODO: do we need to subscribe to this? Or can we pick speed? Assuming we pick for now
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)

    def trajectory_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        print "Receiving new trajectory:", len(msg.poses), "points"
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)
        self.trajectory_points = np.array(msg.points)

    def odom_callback(self, msg): # assuming we get result of localization
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = tf.transformations.euler_from_quaternion(transform.pose.orientation)
        theta = orientation[2]
        self.car_point = (x,y)
        self.pure_pursuit((x, y), theta)
    

    def closest_point_on_line(p1, p2):
        # arguments: two points that define line (a, b), then car_point
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = self.car_point
        
        dx, dy = x2-x1, y2-y1
        det = dx*dx + dy*dy
        assert det != 0
        a = (dy*(y3-y1)+dx*(x3-x1))/det
        
        return x1+a*dx, y1+a*dy 


    def pure_pursuit(self, theta): # not sure what the arguments should be yet
        car_x = self.car_point[0]
        car_y = self.car_point[1]
        dist = lambda p: np.sqrt((car_x - p[0])**2 + (car_y - p[1])**2)
        diff = lambda p: p[0] - car_x, p[1] - car_y
        angle = lambda p: np.arctan2(p[0], p[1])

        # find the differences between the x and y values of current location and points
        points_diff = np.apply_along_axis(diff, self.trajectory_points)
        
        # find relative angle of car to all points
        angles = np.apply_along_axis(angle, points_diff)
        
        # pick out points that are in front of us (-90 < angle < 90 degrees)
        front_trajectory_points = np.any(abs(angles) < np.pi/2)
        angles_idx = np.where(angles == front_trajectory_points)
        front_points = self.trajectory_points[angles_idx]

        # find closest point in front of the car
        point_distances = np.apply_along_axis(dist, front_points, 1)
        closest_point = np.minimum(point_distances)
        idx = np.where(point_distances == closest_point)
        (point_x, point_y) = self.trajectory_points[idx]
        (relative_x, relative_y) = self.closest_point_on_line(point_x, point_y self.car_point)
    
        distance_to_point = np.sqrt(relative_x**2 + relative_y**2)

        rospy.loginfo("Car pose: (%f, %f), Closest point: (%f, %f)", car_x, car_y, point_x, point_y)
        rospy.loginfo("Distance to closest point: %f", distance_to_point)

        alpha = np.arctan2(relative_x, relative_y)
        self.drive_cmd.steering_angle = np.arctan2(2 * self.wheelbase_length * np.sin(alpha), distance_to_point)

        self.drive_pub.publish(self.drive_cmd)


if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
