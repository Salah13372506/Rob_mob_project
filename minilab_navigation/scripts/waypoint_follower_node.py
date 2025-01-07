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
        
        # Paramètres géométriques du robot
        self.l1 = rospy.get_param('~l1', 0.2)  # Distance du point de contrôle à l'axe des roues
        
        # Gains de commande (selon le cours)
        self.k1 = rospy.get_param('~k1', 1.0)  # Gain pour erreur en x
        self.k2 = rospy.get_param('~k2', 1.0)  # Gain pour erreur en y
        
        # Paramètres de sécurité et seuils
        self.max_linear_speed = rospy.get_param('~max_linear_speed', 0.5)
        self.max_angular_speed = rospy.get_param('~max_angular_speed', 1.0)
        self.dist_threshold = rospy.get_param('~dist_threshold', 0.3)
        self.debug = rospy.get_param('~debug', True)
        
        # Variables d'état
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.total_points = 0
        self.is_active = False
        
        # Setup TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.markers_pub = rospy.Publisher('/waypoint_markers', MarkerArray, queue_size=1)
        self.goal_reached_pub = rospy.Publisher('/waypoint/goal_reached', Bool, queue_size=1)
        
        # Subscriber
        rospy.Subscriber('/planned_path', Path, self.path_callback)
        
        # Timer de contrôle (10Hz)
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("Waypoint follower initialized")

    def compute_control(self, x, y, theta, goal_x, goal_y):
        """
        Calcule les commandes selon les équations du cours pour un robot unicycle
        """
        from math import cos, sin
        # 1. Calcul des erreurs dans le repère monde
        ex = x - goal_x
        ey = y - goal_y
        
        # 2. Calcul des vitesses dans le repère monde selon le cours
        v1 = -self.k1 * ex
        v2 = -self.k2 * ey
        
        # 3. Transformation des vitesses monde en vitesses robot
        # Matrice de transformation selon le cours:
        # [v1] = [cos(θ)    -l1*sin(θ)] [u1]
        # [v2]   [sin(θ)     l1*cos(θ)] [u2]
        
        # Inverse de la matrice (calcul analytique pour efficacité)
        u1 = (cos(theta) * v1 + sin(theta) * v2)
        u2 = (-sin(theta) * v1 / self.l1 + cos(theta) * v2 / self.l1)
        
        # 4. Saturation des vitesses
        u1 = np.clip(u1, -self.max_linear_speed, self.max_linear_speed)
        u2 = np.clip(u2, -self.max_angular_speed, self.max_angular_speed)
        
        # 5. Calcul de la distance pour le critère d'arrêt
        distance = np.sqrt(ex**2 + ey**2)
        
        return u1, u2, distance

    def get_robot_pose(self):
        """Obtient la pose actuelle du robot"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            theta = np.arctan2(2.0*(q.w*q.z + q.x*q.y),
                             1.0 - 2.0*(q.y*q.y + q.z*q.z))
            return x, y, theta
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get robot pose: {e}")
            return None

    def path_callback(self, msg):
        """Callback pour recevoir le chemin planifié"""
        self.waypoints = [(pose.pose.position.x, pose.pose.position.y) 
                         for pose in msg.poses]
        self.current_waypoint_idx = 0
        self.total_points = len(self.waypoints)
        self.is_active = True
        self.publish_markers()
        
        if self.debug:
            rospy.loginfo(f"Received path with {self.total_points} waypoints")

    def control_loop(self, event):
        """Boucle principale de contrôle"""
        if not self.is_active or not self.waypoints:
            return
            
        # 1. Obtenir la pose actuelle du robot
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return
            
        x, y, theta = robot_pose
        
        # 2. Obtenir le point cible actuel
        goal_x, goal_y = self.waypoints[self.current_waypoint_idx]
        
        # 3. Calculer les commandes
        v, omega, distance = self.compute_control(x, y, theta, goal_x, goal_y)
        
        # 4. Vérifier si on a atteint le point courant
        if distance < self.dist_threshold:
            self.current_waypoint_idx += 1
            
            # Si on a atteint le dernier point
            if self.current_waypoint_idx >= len(self.waypoints):
                self.is_active = False
                self.stop_robot()
                self.goal_reached_pub.publish(Bool(True))
                rospy.loginfo("Navigation completed")
                return
        
        # 5. Publier les commandes
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        """Arrête le robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def publish_markers(self):
        """Publie les marqueurs de visualisation"""
        marker_array = MarkerArray()
        
        # Marqueur pour les points
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
        
        # Marqueur pour les lignes
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

if __name__ == '__main__':
    try:
        follower = WaypointFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass