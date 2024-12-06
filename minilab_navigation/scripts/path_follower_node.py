#!/usr/bin/env python3
import sys
print("Python path:", sys.path)
print("Python version:", sys.version)
import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from scipy.interpolate import CubicSpline
import tf2_ros
from scipy.optimize import minimize_scalar

class PathFollower:
    def __init__(self):
        # Paramètres du contrôleur
        self.k1 = 2.0  # Gain pour l'erreur d'orientation
        self.k2 = 2.0 # Gain pour l'erreur latérale
        self.v_const = 0.2  # Vitesse linéaire constante
        self.lookahead = 0.5  # Distance de lookahead pour la recherche de point
        
        # Variables pour le chemin
        self.path_x = None
        self.path_y = None
        self.spline_x = None
        self.spline_y = None
        self.current_s = 0.0  # Position curviligne actuelle
        self.path_length = None
        
        # Setup ROS
        rospy.init_node('path_follower')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.path_sub = rospy.Subscriber('/planned_path', Path, self.path_callback)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Timer pour le contrôle
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_callback)

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            return

        # Extraire les points du chemin
        path_points = [(pose.pose.position.x, pose.pose.position.y) 
                      for pose in msg.poses]
        self.path_x = [p[0] for p in path_points]
        self.path_y = [p[1] for p in path_points]
        
        # Créer le paramétrage par distance cumulée
        t = [0]
        for i in range(1, len(self.path_x)):
            dx = self.path_x[i] - self.path_x[i-1]
            dy = self.path_y[i] - self.path_y[i-1]
            t.append(t[-1] + np.sqrt(dx**2 + dy**2))
        
        # Créer les splines
        self.spline_x = CubicSpline(t, self.path_x)
        self.spline_y = CubicSpline(t, self.path_y)
        self.path_length = t[-1]
        self.current_s = 0.0

    def get_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Extraire l'angle depuis le quaternion
            q = transform.transform.rotation
            theta = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 
                             1.0 - 2.0*(q.y*q.y + q.z*q.z))
            
            return x, y, theta
        except:
            rospy.logwarn("Could not get robot pose")
            return None

    def find_closest_point(self, robot_x, robot_y):
        def distance_squared(s):
            path_x = self.spline_x(s)
            path_y = self.spline_y(s)
            return (path_x - robot_x)**2 + (path_y - robot_y)**2
        
        # Recherche locale autour de la position curviligne actuelle
        window = 0.5  # Taille de la fenêtre de recherche
        s_min = max(0, self.current_s - window)
        s_max = min(self.path_length, self.current_s + window)
        
        result = minimize_scalar(distance_squared, 
                               bounds=(s_min, s_max),
                               method='bounded')
        
        return result.x

    def compute_control(self, robot_x, robot_y, robot_theta, s):
        # Position sur le chemin
        path_x = self.spline_x(s)
        path_y = self.spline_y(s)
        
        # Tangente
        dx = self.spline_x.derivative(1)(s)
        dy = self.spline_y.derivative(1)(s)
        path_theta = np.arctan2(dy, dx)
        
        # Erreurs
        theta_e = self.normalize_angle(robot_theta - path_theta)
        d = np.sign((path_y - robot_y)*np.cos(robot_theta) - 
                   (path_x - robot_x)*np.sin(robot_theta)) * \
            np.sqrt((path_x - robot_x)**2 + (path_y - robot_y)**2)
        
        # Loi de commande
        omega = -self.k1*self.v_const*np.sin(theta_e)/theta_e*theta_e - \
                self.k2*self.v_const*d
        
        return self.v_const, omega

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle

    def control_callback(self, event):
        if self.spline_x is None or self.spline_y is None:
            return
            
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
            
        robot_x, robot_y, robot_theta = robot_pose
        
        # Trouver le point le plus proche sur le chemin
        self.current_s = self.find_closest_point(robot_x, robot_y)
        
        # Si on est à la fin du chemin, arrêter
        if self.current_s >= self.path_length - 0.1:
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
            
        # Calculer et publier la commande
        v, omega = self.compute_control(robot_x, robot_y, robot_theta, 
                                      self.current_s)
        
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_vel_pub.publish(cmd)

if __name__ == '__main__':
    try:
        follower = PathFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass