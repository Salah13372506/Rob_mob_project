#!/usr/bin/env python3
import sys
import os
from minilab_navigation.srv import ReturnToHome, ReturnToHomeResponse 
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from astar_planner import AStarPlanner
import rospy
from nav_msgs.msg import Path
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import tf2_ros
import tf2_geometry_msgs

class PathPlannerNode:
    def __init__(self):
        rospy.init_node('path_planner_node')
        
        self.debug = True
        self.planner = AStarPlanner()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Position de départ par défaut
        self.start_pose = PoseStamped()
        self.start_pose.pose.position.x = 0
        self.start_pose.pose.position.y = 0
        self.home_position = None
        
        # Publishers
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1)
        self.marker_pub = rospy.Publisher('/path_markers', Marker, queue_size=1)
        
        # Attendre que le service dynamic_map soit disponible
        rospy.wait_for_service('dynamic_map')
        self.get_map = rospy.ServiceProxy('dynamic_map', GetMap)
        
        # Subscriber pour le goal
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        # Service de retour à la base
        self.return_home_service = rospy.Service('return_to_home', ReturnToHome, self.handle_return_home)
        
        # Enregistrer la position initiale
        self.record_home_position()

        if self.debug:
            rospy.loginfo("Path planner node initialized")


    def record_home_position(self):
        """Enregistre la position actuelle comme position de base"""
        rospy.sleep(1.0)  # Attendre que TF soit prêt
        try:
            pos = self.get_robot_pose()
            self.home_position = pos
            rospy.loginfo(f"Home position recorded at: {pos}")
        except Exception as e:
            rospy.logwarn(f"Could not record home position: {e}")
            self.home_position = (0, 0)  # Position par défaut

    def handle_return_home(self, req):
        """Gestionnaire du service de retour à la base"""
        if self.home_position is None:
            return ReturnToHomeResponse(False, "Home position not set")
        
        if not self.update_map():
            return ReturnToHomeResponse(False, "Failed to get map from service")

        try:
            # Obtenir la position actuelle
            current_pos = self.get_robot_pose()
            
            # Calculer le chemin vers la position de base
            path = self.planner.find_path(
                current_pos,
                self.home_position
            )
            
            if path:
                self.publish_path(path)
                self.publish_markers(path)
                return ReturnToHomeResponse(True, "Path to home published")
            else:
                return ReturnToHomeResponse(False, "Could not find path to home")
                
        except Exception as e:
            return ReturnToHomeResponse(False, f"Error: {str(e)}")


    def update_map(self):
        """Récupère la carte via le service dynamic_map"""
        try:
            response = self.get_map()
            if response is not None:
                self.planner.set_map(response.map)
                if self.debug:
                    rospy.loginfo("Successfully got map from service")
                return True
            return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def get_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            if self.debug:
                rospy.loginfo(f"Robot position: {transform.transform.translation.x}, {transform.transform.translation.y}")
            return (transform.transform.translation.x, transform.transform.translation.y)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get robot pose: {e}")
            if self.debug:
                rospy.loginfo("Using default start position (0,0)")
            return (0, 0)

    def goal_callback(self, goal_msg):
        if self.debug:
            rospy.loginfo(f"Received goal: {goal_msg.pose.position.x}, {goal_msg.pose.position.y}")
        
        # Obtenir la carte mise à jour avant de planifier
        if not self.update_map():
            rospy.logerr("Failed to get map from service, cannot plan path")
            return
        
        # Obtenir la position actuelle du robot
        start_pos = self.get_robot_pose()
        
        # Calculer le chemin
        path = self.planner.find_path(
            start_pos,
            (goal_msg.pose.position.x, goal_msg.pose.position.y)
        )

        if path:
            if self.debug:
                rospy.loginfo(f"Path found with {len(path)} points")
            self.publish_path(path)
            self.publish_markers(path)
        else:
            rospy.logwarn("No path found!")

    def publish_path(self, path):
        """Publie le chemin comme un message Path"""
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = rospy.Time.now()

        for x, y in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def publish_markers(self, path):
        """Publie des marqueurs de visualisation pour RViz"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for x, y in path:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.1
            marker.points.append(point)

        self.marker_pub.publish(marker)

if __name__ == '__main__':
    try:
        node = PathPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass