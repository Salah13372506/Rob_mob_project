#!/usr/bin/env python3
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool

class WaypointFollower:
    def __init__(self):
        rospy.init_node('waypoint_follower')
        
        # Paramètres du contrôleur
        self.k_linear = rospy.get_param('~k_linear', 1)    # Gain proportionnel vitesse linéaire (réduit)
        self.k_angular = rospy.get_param('~k_angular', 5)  # Gain proportionnel vitesse angulaire (réduit)
        self.k_smooth = rospy.get_param('~k_smooth', 0.2)    # Facteur de lissage pour la vitesse
        self.prev_v = 0.0  # Pour le lissage de la vitesse linéaire
        self.prev_omega = 0.0  # Pour le lissage de la vitesse angulaire
        
        # Seuils et paramètres de sécurité
        self.dist_threshold = rospy.get_param('~dist_threshold', 0.1)
        self.max_linear_speed = rospy.get_param('~max_linear_speed', 1.5)
        self.max_angular_speed = rospy.get_param('~max_angular_speed', 1.0)
        self.debug = rospy.get_param('~debug', True)

        self.total_points = 0
        self.current_waypoint_idx = 0
        
        # Variables d'état
        self.waypoints = []
        self.is_active = False
        
        # Setup TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.markers_pub = rospy.Publisher('/waypoint_markers', MarkerArray, queue_size=1)
        self.goal_reached_pub = rospy.Publisher('/waypoint/goal_reached', Bool, queue_size=1)
        
        # Paramètres de navigation
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 0.2)
        self.max_points_ahead = rospy.get_param('~max_points_ahead', 5)

        # Subscriber pour le chemin planifié
        rospy.Subscriber('/planned_path', Path, self.path_callback)
        
        # Timer de contrôle
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("Waypoint follower initialized")

    def find_best_waypoint(self, x, y, current_idx):
        """Trouve le meilleur point à suivre en regardant plusieurs points en avant"""
        best_distance = float('inf')
        best_idx = current_idx
        
        end_idx = min(current_idx + self.max_points_ahead, len(self.waypoints))
        
        for idx in range(current_idx, end_idx):
            wp_x, wp_y = self.waypoints[idx]
            distance = np.sqrt((wp_x - x)**2 + (wp_y - y)**2)
            
            if distance < self.lookahead_distance:
                best_idx = idx + 1
                if best_idx >= len(self.waypoints):
                    return len(self.waypoints) - 1
            
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
                
        return best_idx

    def compute_control(self, x, y, theta, goal_x, goal_y):
        """Calcule les commandes avec une loi proportionnelle améliorée"""
        # Calcul des erreurs
        dx = goal_x - x
        dy = goal_y - y
        
        # Distance euclidienne au but
        distance = np.sqrt(dx**2 + dy**2)
        
        # Angle désiré vers le but
        desired_theta = np.arctan2(dy, dx)
        
        # Erreur d'angle
        theta_error = self.normalize_angle(desired_theta - theta)
        
        # Calcul de la vitesse linéaire avec une fonction non-linéaire
        # Utilisation d'une fonction exponentielle pour avoir une approche plus douce
        v_target = self.k_linear * distance * np.exp(-abs(theta_error))
        
        # La vitesse angulaire avec terme proportionnel à la distance
        omega_target = self.k_angular * theta_error
        
        # Ajout d'un terme de correction pour mieux suivre la ligne
        if distance > self.dist_threshold:
            omega_target += self.k_angular * 0.5 * np.sign(theta_error) * (distance / self.lookahead_distance)
        
        # Lissage des vitesses pour éviter les changements brusques
        v = (1 - self.k_smooth) * self.prev_v + self.k_smooth * v_target
        omega = (1 - self.k_smooth) * self.prev_omega + self.k_smooth * omega_target
        
        # Saturation des vitesses
        v = np.clip(v, -self.max_linear_speed, self.max_linear_speed)
        omega = np.clip(omega, -self.max_angular_speed, self.max_angular_speed)
        
        # Enregistrement des vitesses pour le prochain cycle
        self.prev_v = v
        self.prev_omega = omega
        
        return v, omega, distance

    def normalize_angle(self, angle):
        """Normalise un angle dans [-pi, pi]"""
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle

    def path_callback(self, msg):
        """Callback pour recevoir le chemin planifié"""
        self.waypoints = [(pose.pose.position.x, pose.pose.position.y) 
                         for pose in msg.poses]
        self.current_waypoint_idx = 0
        self.total_points = len(self.waypoints)
        self.is_active = True
        
        rospy.loginfo("=== Nouveau chemin reçu ===")
        rospy.loginfo(f"Nombre total de points: {self.total_points}")
        if self.debug and self.waypoints:
            rospy.loginfo("Points de passage:")
            for i, (x, y) in enumerate(self.waypoints):
                rospy.loginfo(f"  Point {i}: ({x:.2f}, {y:.2f})")
        
        self.publish_markers()

    def get_robot_pose(self):
        """Obtient la pose actuelle du robot"""
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link',
                                                      rospy.Time(0), 
                                                      rospy.Duration(1.0))
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            theta = np.arctan2(2.0*(q.w*q.z + q.x*q.y),
                             1.0 - 2.0*(q.y*q.y + q.z*q.z))
            return x, y, theta
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get robot pose: {e}")
            return None

    def publish_markers(self):
        """Publie les marqueurs de visualisation des points de passage"""
        marker_array = MarkerArray()
        
        points_marker = Marker()
        points_marker.header.frame_id = "map"
        points_marker.header.stamp = rospy.Time.now()
        points_marker.ns = "waypoints"
        points_marker.id = 0
        points_marker.type = Marker.POINTS
        points_marker.action = Marker.ADD
        points_marker.scale.x = 0.1
        points_marker.scale.y = 0.1
        points_marker.color.r = 1.0
        points_marker.color.a = 1.0
        
        lines_marker = Marker()
        lines_marker.header.frame_id = "map"
        lines_marker.header.stamp = rospy.Time.now()
        lines_marker.ns = "waypoints_lines"
        lines_marker.id = 1
        lines_marker.type = Marker.LINE_STRIP
        lines_marker.action = Marker.ADD
        lines_marker.scale.x = 0.03
        lines_marker.color.g = 1.0
        lines_marker.color.a = 0.5
        
        for x, y in self.waypoints:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.1
            points_marker.points.append(p)
            lines_marker.points.append(p)
            
        marker_array.markers.append(points_marker)
        marker_array.markers.append(lines_marker)
        self.markers_pub.publish(marker_array)

    def control_loop(self, event):
        """Boucle principale de contrôle"""
        if not self.is_active or not self.waypoints:
            return
            
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
            
        x, y, theta = robot_pose
        self.current_waypoint_idx = self.find_best_waypoint(x, y, self.current_waypoint_idx)
        goal_x, goal_y = self.waypoints[self.current_waypoint_idx]
        
        v, omega, distance = self.compute_control(x, y, theta, goal_x, goal_y)
        
        if self.current_waypoint_idx >= len(self.waypoints) - 1 and distance < self.dist_threshold:
            self.is_active = False
            self.stop_robot()
            self.goal_reached_pub.publish(Bool(True))
            rospy.loginfo("=== Navigation terminée ===")
            return
            
        if self.debug and rospy.Time.now().to_sec() % 5 < 0.1:
            progress = (self.current_waypoint_idx / self.total_points) * 100
            rospy.loginfo(f"Progression: {self.current_waypoint_idx+1}/{self.total_points} " +
                         f"(distance: {distance:.2f}m, {progress:.1f}%)")
        
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        """Arrête le robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

if __name__ == '__main__':
    try:
        follower = WaypointFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
