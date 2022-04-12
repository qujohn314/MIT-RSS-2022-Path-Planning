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
        self.lookahead = 3  # TODO: Refine this number; change with trajectory to optimize as well
        self.speed = 1  # TODO: Any changes needed? Do we need to get speed as parameter?
        self.wheelbase_length = 1.5
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

    def odom_callback(self, msg):
        # rospy.loginfo("ODOM CALLBACK -------------------")
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        rospy.loginfo("CAR POSITION ---------------")
        rospy.loginfo(x)
        rospy.loginfo(y)
        quat = msg.pose.pose.orientation
        orientation = tf.transformations.euler_from_quaternion(np.array([quat.x, quat.y, quat.z, quat.w]))
        theta = orientation[2]

        self.car_theta = theta
        self.car_point = (x, y)
        self.pure_pursuit()

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

    def pure_pursuit(self):
        if len(self.trajectory.points) == 0:
            print("Have not received trajectory yet...")
            return
        
        car_x = self.car_point[0]
        car_y = self.car_point[1]
        dist = lambda p: np.sqrt((self.car_point[0] - p[0])**2 + (self.car_point[1] - p[1])**2)
        angle = lambda p: np.arctan2(p[1], p[0])


        # find the differences between the x and y values of current location and points
        points_diff = np.apply_along_axis(self.diff, 1, self.trajectory.points)
        points_dist = np.apply_along_axis(dist, 0, points_diff)
        # rospy.loginfo("POINTS DIFF ----------------------------")
        # rospy.loginfo(points_diff)

        # find relative angle of car to all points
        # rospy.loginfo("CAR THETA -----------------------")
        # rospy.loginfo(self.car_theta)
        angles = np.apply_along_axis(angle, 0, points_dist)
        # rospy.loginfo("ANGLES ----------------------------")
        # rospy.loginfo(angles)

        
        # pick out points that are in front of us (-90 < angle < 90 degrees)
        front_trajectory_points = np.where(np.abs(angles) < np.pi/2.0)[0]
        # rospy.loginfo("FRONT TRAJ IDX ----------------------------")
        # rospy.loginfo(front_trajectory_points)

        traj_pts_arr = np.array(self.trajectory.points)
        front_points = traj_pts_arr[front_trajectory_points]

        # find closest point in front of the car
        point_distances = np.apply_along_axis(dist, 0, front_points)
        closest_point = np.amin(point_distances)
        idx = np.where(point_distances == closest_point)[0]

        # rospy.loginfo("FRONT TRAJ IDX ----------------------------")
        # rospy.loginfo(traj_pts_arr)
        # rospy.loginfo(traj_pts_arr[idx])
        # rospy.loginfo(idx)
        (point_x, point_y) = (traj_pts_arr[idx][0][0], traj_pts_arr[idx][0][1])
        (point2_x, point2_y) = (traj_pts_arr[idx+1][0][0], traj_pts_arr[idx+1][0][1])
        (relative_x, relative_y) = self.closest_point_on_line((point_x, point_y), (point2_x, point2_y))
    
        distance_to_point = np.sqrt((self.car_point[0] - point_x)**2 + (self.car_point[1] - point_y)**2)

        rospy.loginfo("Car pose: (%f, %f), Closest point: (%f, %f)", car_x, car_y, point_x, point_y)
        rospy.loginfo("Distance to closest point: %f", distance_to_point)

        alpha = np.arctan2(relative_x, relative_y)
        rospy.loginfo("ALPHA ---------------------------------------")
        rospy.loginfo(alpha)
        rospy.loginfo(idx)
        angle_cmd = np.arctan2(2 * self.wheelbase_length * np.sin(alpha), distance_to_point)

        if alpha < 0:
            self.drive_cmd.drive.steering_angle = -angle_cmd
        else:
            self.drive_cmd.drive.steering_angle = angle_cmd

        self.drive_pub.publish(self.drive_cmd)


if __name__ == "__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
