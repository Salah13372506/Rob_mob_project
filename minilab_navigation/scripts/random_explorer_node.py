#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import random
import math
import tf2_ros

class SimpleExplorer:
    def __init__(self):
        rospy.init_node('simple_explorer')
        
        # Paramètres de navigation
        self.min_front_distance = rospy.get_param('~min_distance', 1.3)
        self.linear_speed = rospy.get_param('~max_linear_speed', 0.3) 
        self.angular_speed = rospy.get_param('~max_angular_speed', 0.8)
        
        # Variables d'état
        self.current_scan = None
        self.yaw = 0.0
        self.current_position = [0, 0]
        self.is_turning = False
        self.turn_direction = 0
        
        # TF Buffer setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.turn_start_time = None
        self.turn_duration = rospy.Duration(2.0)  
    
        # Suivi de trajectoire
        self.path_points = []
        
        # Publishers et Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.path_pub = rospy.Publisher('/robot_path', Marker, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # Timer pour le contrôle et la mise à jour de la position
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        self.position_timer = rospy.Timer(rospy.Duration(0.1), self.update_position)
        
        rospy.loginfo("Simple explorer initialized")

    def update_position(self, event):
        """Met à jour la position du robot en utilisant TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            
            # Mise à jour de la position
            self.current_position[0] = transform.transform.translation.x
            self.current_position[1] = transform.transform.translation.y
            
            # Calcul du yaw à partir du quaternion
            q = transform.transform.rotation
            _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            
            # Ajout d'un point au chemin si déplacement significatif
            if not self.path_points or self.distance_to_last_point() > 0.1:
                self.path_points.append(self.current_position.copy())
                self.publish_path()
                
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get robot pose: {e}")

    def publish_path(self):
        """Création et publication du marqueur pour visualiser le chemin"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.05  # Largeur de la ligne
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        for point in self.path_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.1
            marker.points.append(p)
        
        self.path_pub.publish(marker)

    def distance_to_last_point(self):
        """Calcule la distance entre la position actuelle et le dernier point enregistré"""
        if not self.path_points:
            return float('inf')
        last_point = self.path_points[-1]
        return math.sqrt((self.current_position[0] - last_point[0])**2 + 
                        (self.current_position[1] - last_point[1])**2)

    def scan_callback(self, msg):
        """Callback pour les données du scanner laser"""
        self.current_scan = msg

    def get_sector_distance(self, angle_start, angle_end):
        """Calcul de la distance minimale dans un secteur angulaire"""
        if not self.current_scan:
            return float('inf')
            
        start_idx = int((angle_start - self.current_scan.angle_min) / self.current_scan.angle_increment)
        end_idx = int((angle_end - self.current_scan.angle_min) / self.current_scan.angle_increment)
        start_idx = max(0, min(start_idx, len(self.current_scan.ranges) - 1))
        end_idx = max(0, min(end_idx, len(self.current_scan.ranges) - 1))
        
        valid_distances = []
        for i in range(start_idx, end_idx):
            if self.current_scan.range_min <= self.current_scan.ranges[i] <= self.current_scan.range_max:
                valid_distances.append(self.current_scan.ranges[i])
                
        return min(valid_distances) if valid_distances else float('inf')

    def get_distances(self):
        """Obtention des distances pour chaque secteur"""
        return {
            'front': self.get_sector_distance(-0.5, 0.5),
            'left': self.get_sector_distance(0.5, 1.5),
            'right': self.get_sector_distance(-1.5, -0.5)
        }

    def normalize_angle(self, angle):
        """Normalisation de l'angle entre -pi et pi"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def control_loop(self, event):
        """Boucle principale de contrôle du robot"""
        if not self.current_scan:
            return
            
        cmd = Twist()
        distances = self.get_distances()
        rospy.loginfo(f"Distances - Front: {distances['front']:.2f}, Left: {distances['left']:.2f}, Right: {distances['right']:.2f}")
        
        if self.is_turning:
            error = self.normalize_angle(self.target_yaw - self.yaw)
            
            if abs(error) < 0.1:
                self.is_turning = False
                cmd.angular.z = 0.0
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_speed * (-self.turn_direction)
        
        elif distances['front'] < self.min_front_distance:
            # Choix de la direction de rotation selon les distances latérales
            self.turn_direction = -1 if distances['right'] < distances['left'] else 1
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed * (-self.turn_direction)
        
        else:
            cmd.linear.x = self.linear_speed
            # Léger ajustement pour éviter les obstacles latéraux
            if distances['left'] < self.min_front_distance * 1.5:
                cmd.angular.z = -0.2
            elif distances['right'] < self.min_front_distance * 1.5:
                cmd.angular.z = 0.2
            else:
                cmd.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd)

if __name__ == '__main__':
    try:
        explorer = SimpleExplorer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
