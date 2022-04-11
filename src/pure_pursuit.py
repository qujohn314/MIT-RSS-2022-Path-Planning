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
        # self.odom_topic = rospy.get_param("~odom_topic")  # TODO: This is throwing an error for some reason
        self.odom_topic = "/odom"
        self.lookahead = 1  # TODO: Refine this number; change with trajectory to optimize as well
        self.speed = 1  # TODO: Any changes needed? Do we need to get speed as parameter?
        self.wheelbase_length = 0.55
        self.trajectory = utils.LineTrajectory("/followed_trajectory")
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)
        self.drive_cmd = AckermannDriveStamped()
        self.drive_cmd.drive.speed = self.speed  # TODO: do we need to subscribe to this? Or can we pick speed?
        # TODO Assuming we pick for now


    def trajectory_callback(self, msg):
        """
        Clears the currently followed trajectory, and loads the new one from the message
        """
        # print "Receiving new trajectory:", len(msg.poses), "points"
        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

    def odom_callback(self, msg):  # assuming we get result of localization
        # rospy.loginfo("ODOM CALLBACK -------------------")
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        orientation = tf.transformations.euler_from_quaternion(np.array([quat.x, quat.y, quat.z, quat.w]))
        theta = orientation[2]
        self.car_point = (x, y)
        self.pure_pursuit(theta)

    def closest_point_on_line(self, p1, p2):
        # arguments: two points that define line (a, b), then car_point
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = self.car_point

        dx, dy = x2-x1, y2-y1
        det = dx*dx + dy*dy
        assert det != 0
        a = (dy*(y3-y1)+dx*(x3-x1))/det
        
        return x1+a*dx, y1+a*dy

    def diff(self, point):
        point[0] = point[0] - self.car_point[0]
        point[1] = point[1] - self.car_point[1]
        return point

    def pure_pursuit(self, theta):  # not sure what the arguments should be yet
        if len(self.trajectory.points) == 0:
            print("Have not received trajectory yet...")
            return
        
        car_x = self.car_point[0]
        car_y = self.car_point[1]
        dist = lambda p: np.sqrt((car_x - p[0])**2 + (car_y - p[1])**2)
        angle = lambda p: np.arctan2(p[1], p[0])

        # find the differences between the x and y values of current location and points
        points_diff = np.apply_along_axis(self.diff, 0, self.trajectory.points)
        
        # find relative angle of car to all points
        # rospy.loginfo()
        angles = np.apply_along_axis(angle, 0, points_diff)
        
        # pick out points that are in front of us (-90 < angle < 90 degrees)
        front_trajectory_points = np.any(abs(angles) < np.pi/2)
        angles_idx = np.where(angles == front_trajectory_points)
        rospy.loginfo("ANGLES IDX ----------------------------")
        rospy.loginfo(angles_idx)
        front_points = self.trajectory.points[angles_idx]

        # find closest point in front of the car
        point_distances = np.apply_along_axis(dist, 0, front_points)
        closest_point = np.minimum(point_distances)
        idx = np.where(point_distances == closest_point)
        (point_x, point_y) = self.trajectory.points[idx]
        (relative_x, relative_y) = self.closest_point_on_line(point_x, point_y)
    
        distance_to_point = np.sqrt(relative_x**2 + relative_y**2)

        rospy.loginfo("Car pose: (%f, %f), Closest point: (%f, %f)", car_x, car_y, point_x, point_y)
        rospy.loginfo("Distance to closest point: %f", distance_to_point)

        alpha = np.arctan2(relative_x, relative_y)
        self.drive_cmd.steering_angle = np.arctan2(2 * self.wheelbase_length * np.sin(alpha), distance_to_point)

        self.drive_pub.publish(self.drive_cmd)


if __name__ == "__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
