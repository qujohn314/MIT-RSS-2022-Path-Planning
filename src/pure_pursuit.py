#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils
import tf

import sys


from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry


class PurePursuit(object):
    """ 
    Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        # self.odom_topic = rospy.get_param("~odom_topic")  # TODO: This is throwing an error for some reason
        self.odom_topic = "/odom"
        self.standard_lookahead = 1  # TODO: Refine this number; change with trajectory to optimize as well
        self.lookahead_distance = 1  # lookahead distance currently being used; scaled in pure_pursuit() based on curvature of trajectory
        self.speed = 1  # TODO: Any changes needed? Do we need to get speed as parameter?
        self.wheelbase_length = 0.8  #TODO is this okay? 
        self.trajectory = utils.LineTrajectory("/followed_trajectory")
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.traj_sub = rospy.Subscriber("/trajectory/current", PoseArray, self.trajectory_callback, queue_size=1)
        self.drive_pub = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)
        self.drive_cmd = AckermannDriveStamped()
        self.drive_cmd.drive.speed = self.speed  # TODO: do we need to subscribe to this? Or can we pick speed?
        self.points_list = np.array([])  # stores three tuples (x,y) corresponding to the three points we're using
        self.points_dist = np.array([])
        self.midpoint_buffer  = 0.5  # if the car is within this distance of the midpoint, move to the next line segment
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
        # rospy.loginfo("CAR POSITION ---------------")
        # rospy.loginfo(x)
        # rospy.loginfo(y)
        quat = msg.pose.pose.orientation
        orientation = tf.transformations.euler_from_quaternion(np.array([quat.x, quat.y, quat.z, quat.w]))
        theta = orientation[2]

        self.car_theta = theta
        self.car_point = (x, y)
        self.pure_pursuit()
        
    def closest_point_to_lookahead(self, pt1, pt2, tangent_tol=1e-9):
        """ Find the points at which a circle intersects a line-segment.
        Returns p2 if no intersection is found, otherwise returns intersection that is closest to p2.
        """
        distance_from_pt2 = lambda p: ((pt2[0] - p[0])**2 + (pt2[1] - p[1])**2)**0.5

        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, self.car_point
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
        dx, dy = (x2 - x1), (y2 - y1)
        dr = (dx ** 2 + dy ** 2)**.5
        big_d = x1 * y2 - x2 * y1
        discriminant = self.lookahead_distance ** 2 * dr ** 2 - big_d ** 2

        if discriminant < 0:  # No intersection between circle and line
            rospy.loginfo("No intersection between lookahead circle and current segment...")
            rospy.loginfo(self.points_idx)
            return pt1
        else:  # There may be 0, 1, or 2 intersections with the segment
            intersections = [
                (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2, cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2) for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
                # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
            if len(intersections) >= 1 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
                return intersections[0]
            else:
                if len(intersections) == 0:
                    rospy.loginfo("No intersection between lookahead circle and current segment...HERE")
                    return pt1
                if len(intersections) > 1:
                    assert len(intersections) == 2
                    if distance_from_pt2(intersections[1]) < distance_from_pt2(intersections[0]): # pt closer to pt2
                        return intersections[1]
                    else:
                        return intersections[0]

                return intersections[0]


    def diff(self, point):
        point[0] = point[0] - self.car_point[0]
        point[1] = point[1] - self.car_point[1]
        return point

    def find_segment_idxs(self):
        """
        Adds points to self.points_list: a numpy array of three tuples (x_n, y_n) corresponding to the three points we need to use for pure pursuit
        """
        
        # do we have 3 points?
        # find two closest points to the car (as idx along traj) and pick the one furthest from the start point
        car_x = self.car_point[0]
        car_y = self.car_point[1]
        rospy.loginfo("CAR POSITION ---------------------------------!!!!!!!!!!!!!!!!!!????????????")
        rospy.loginfo(self.car_point)


        dist = lambda p: np.sqrt((self.car_point[0] - p[0])**2 + (self.car_point[1] - p[1])**2)
        # points_diff = np.apply_along_axis(self.diff, 1, self.trajectory.points)
        points_dist = np.apply_along_axis(dist, 1, self.trajectory.points)
        rospy.loginfo("trajectory points --------------")
        rospy.loginfo(self.trajectory.points)

        rospy.loginfo("points dist --------------")
        rospy.loginfo(points_dist)

        
        closest_points_idx = np.argpartition(points_dist, 2)[:2] # get the two closest points

        closest_points_dist = np.array(map(self.trajectory.distance_along_trajectory, closest_points_idx))  
        sorted_idx = np.argsort(-closest_points_dist)
        closest_points_idx = closest_points_idx[np.argsort(-closest_points_dist)] # sort points in descending order, furthest along trajectory
        
        rospy.loginfo("Closest point idx --------------")
        rospy.loginfo(closest_points_idx)

        idx1 = closest_points_idx[0]
        # idx1 = max(0, idx2-1)
        idx2 = min(idx1+1, len(self.trajectory.points)-1)
        idx3 = min(idx2+1, len(self.trajectory.points)-1)

        self.points_idx = np.array([idx1, idx2, idx3])
        rospy.loginfo("point idx --------------")
        rospy.loginfo(self.points_idx)

        self.points_list = np.array(self.trajectory.points)[self.points_idx]
        rospy.loginfo("point list --------------")
        rospy.loginfo(self.points_list)

        

    def curvature(self):
        """
        Returns curvature of line calculated from the three points on the trajectory currently in use
        """
        
        x1 = self.points_list[0][0]
        x2 = self.points_list[1][0]
        x3 = self.points_list[2][0]
        y1 = self.points_list[0][1]
        y2 = self.points_list[1][1]
        y3 = self.points_list[2][1]
        
        ab = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)
        ac = ((x1 - x3)**2 + (y1 - y3)**2)**(1/2)
        bc = ((x2 - x3)**2 + (y2 - y3)**2)**(1/2)

        area = 1/2*(ac)*((ab)**2 - (ac/2)**2)**(1/2)

        curvature = 4*area/(ab * ac * bc)
        return curvature

        
    def update_idx(self):
        rospy.loginfo("UPDATING IDX!!!!!! --------------")
        rospy.loginfo(self.points_idx)

        self.points_idx[0] = self.points_idx[1]
        self.points_idx[1] = self.points_idx[2]
        self.points_idx[2] = min(self.points_idx[2] + 1, len(self.trajectory.points)-1)
        rospy.loginfo("point idx --------------")
        rospy.loginfo(self.points_idx)

        self.points_list = np.array(self.trajectory.points)[self.points_idx]


    def pure_pursuit(self):
        """
        Runs pure pursuit algorithm for navigation. Outputs command for racecar's angular movement
        """
        if len(self.trajectory.points) == 0:
            print("Waiting for trajectory to load...")
            time.sleep(3)
            return
        
        if not hasattr(self, "points_idx") or len(self.points_idx) < 3:
            # Case 0: We don't have three points yet, so we pick them
            print("HERE")
            self.find_segment_idxs()
        
        distance_from_car = lambda p: np.sqrt((self.car_point[0] - p[0])**2 + (self.car_point[1] - p[1])**2)
        dist = lambda p: np.sqrt((self.car_point[0] - p[0])**2 + (self.car_point[1] - p[1])**2)


        self.relative_dist = np.apply_along_axis(distance_from_car, 1, self.points_list)
        # rospy.loginfo("RELATIVE DISTANCES --------------------------------OOOOOOOOOOOOOO")
        # rospy.loginfo(self.relative_dist)

        if self.relative_dist[1] < self.standard_lookahead/2.0:
            self.update_idx()

        # if self.relative_dist[1] > self.standard_lookahead:
            # Case 1: We have three unique points and use a point on line segment 1
        if self.points_idx[1] == self.points_idx[2]:
            # When we get to end of trajectory and don't have three unique points
            curvature = 1

        elif self.curvature() > 10000000:
            # Need to add this in case of curvature = 0, i.e. straight line
            curvature = self.curvature()
        else:
            curvature = 1
            
        self.lookahead_distance = 1/curvature * self.standard_lookahead # scale lookahead distance based on inverse of trajectory curvature           
        pursuit_point = self.closest_point_to_lookahead(self.points_list[0], self.points_list[1]) # point we use for pure pursuit algorithm  
        # pursuit_point = self.points_list[0]
        relative_xy = (self.car_point[0] - pursuit_point[0], self.car_point[1] - pursuit_point[1]) # relative diff in x, y between car and pursuit point           
        distance_to_pursuit_point = np.sqrt(relative_xy[0]**2 + relative_xy[1]**2)  # distance from point to car
        alpha = np.arctan2(relative_xy[1], relative_xy[0]) + self.car_theta # angle between point and car

        angle_cmd = 0.5 * np.arctan2(2 * self.wheelbase_length * np.sin(alpha), distance_to_pursuit_point)
            
        self.drive_cmd.drive.steering_angle = angle_cmd

        self.drive_pub.publish(self.drive_cmd)
    

if __name__ == "__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
