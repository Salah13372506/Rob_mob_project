#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import tf2_ros
import tf2_geometry_msgs
from astar_planner import AStarPlanner 

class PathPlannerNode:
    def __init__(self):
        rospy.init_node('path_planner_node')
        
        # Pour debug
        self.debug = True
        
        self.planner = AStarPlanner()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Position de départ par défaut (centre de la carte)
        self.start_pose = PoseStamped()
        self.start_pose.pose.position.x = 0
        self.start_pose.pose.position.y = 0
        
        # Publishers
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=1)
        self.marker_pub = rospy.Publisher('/path_markers', Marker, queue_size=1)
        
        # Subscribers - Changement du topic pour correspondre à RViz
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        if self.debug:
            rospy.loginfo("Path planner node initialized")

    def map_callback(self, msg):
        if self.debug:
            rospy.loginfo("Received map update")
        self.planner.set_map(msg)

    def get_robot_pose(self):
        try:
            # Essayer d'obtenir la transformation de base_link vers map
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
            # En cas d'échec, utiliser une position par défaut
            if self.debug:
                rospy.loginfo("Using default start position (0,0)")
            return (0, 0)

    def goal_callback(self, goal_msg):
        if self.debug:
            rospy.loginfo(f"Received goal: {goal_msg.pose.position.x}, {goal_msg.pose.position.y}")
        
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
        marker.scale.x = 0.05  # Largeur de la ligne
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