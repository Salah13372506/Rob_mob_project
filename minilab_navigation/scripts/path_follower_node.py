#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
from scipy.interpolate import CubicSpline
import tf2_ros
import tf2_geometry_msgs

class PathFollowerNode:
    def __init__(self):
        rospy.init_node('path_follower_node')
        
        # Paramètres de suivi de trajectoire
        self.look_ahead_distance = rospy.get_param('~look_ahead_distance', 0.3)
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 0.3)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 1.0)
        
        # Variables pour stocker la trajectoire
        self.spline_x = None
        self.spline_y = None
        self.total_length = 0
        self.current_path = None
        
        # Configuration TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publishers et Subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.path_sub = rospy.Subscriber('/planned_path', Path, self.path_callback)
        
        # Timer pour le contrôle
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_callback)
        
        rospy.loginfo("Path follower node initialized")

    def path_callback(self, path_msg):
        """Reçoit le nouveau chemin et crée l'interpolation"""
        if len(path_msg.poses) < 2:
            rospy.logwarn("Path too short for interpolation")
            return
            
        # Extraire les points du chemin
        points = [(pose.pose.position.x, pose.pose.position.y) 
                 for pose in path_msg.poses]
        self.interpolate_path(points)
        self.current_path = path_msg
        rospy.loginfo("New path received and interpolated")

    def interpolate_path(self, points):
        """Crée une spline à partir des points du chemin"""
        # Extraire coordonnées x et y
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Calculer la distance cumulée comme paramètre
        dists = [0]
        for i in range(1, len(points)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            d = np.sqrt(dx*dx + dy*dy)
            dists.append(dists[-1] + d)
        
        # Normaliser les distances entre 0 et 1
        self.total_length = dists[-1]
        t = [d/self.total_length for d in dists]
        
        # Créer les splines
        self.spline_x = CubicSpline(t, x_coords)
        self.spline_y = CubicSpline(t, y_coords)

    def get_robot_pose(self):
        """Obtient la position actuelle du robot"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            return (transform.transform.translation.x, 
                    transform.transform.translation.y)
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get robot pose: {e}")
            return None

    def find_closest_point(self, robot_pose):
        """Trouve le point le plus proche sur la spline"""
        if self.spline_x is None or robot_pose is None:
            return None
            
        # Échantillonner la spline pour trouver le point le plus proche
        t_samples = np.linspace(0, 1, 100)
        min_dist = float('inf')
        closest_t = 0
        
        for t in t_samples:
            x = self.spline_x(t)
            y = self.spline_y(t)
            dist = np.sqrt((x - robot_pose[0])**2 + (y - robot_pose[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_t = t
                
        return closest_t

    def compute_control(self, robot_pose, t_closest):
        """Calcule les commandes de vitesse"""
        # Calcule le point cible sur la trajectoire
        t_target = min(1.0, t_closest + self.look_ahead_distance/self.total_length)
        target_x = self.spline_x(t_target)
        target_y = self.spline_y(t_target)
        
        # Calcule l'erreur d'orientation
        dx = target_x - robot_pose[0]
        dy = target_y - robot_pose[1]
        target_angle = np.arctan2(dy, dx)
        
        # Crée la commande
        cmd = Twist()
        cmd.linear.x = self.max_linear_vel
        angle_diff = target_angle
        cmd.angular.z = np.clip(angle_diff, -self.max_angular_vel, self.max_angular_vel)
        
        return cmd

    def control_callback(self, event):
        """Callback du timer pour le contrôle"""
        if self.current_path is None or self.spline_x is None:
            return
            
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
            
        t_closest = self.find_closest_point(robot_pose)
        if t_closest is not None:
            cmd = self.compute_control(robot_pose, t_closest)
            self.cmd_pub.publish(cmd)

if __name__ == '__main__':
    try:
        node = PathFollowerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass