#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class TeleopNode:
    def __init__(self):
        rospy.init_node('minilab_teleop')
        
        # Paramètres
        self.linear_scale = rospy.get_param('~linear_scale', 10.0)  # m/s
        self.angular_scale = rospy.get_param('~angular_scale', 3.0)  # rad/s
        
        # Publishers et Subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback)
        
        self.cmd_msg = Twist()
        
    def joy_callback(self, joy_msg):
        # Axe vertical gauche pour vitesse linéaire
        self.cmd_msg.linear.x = joy_msg.axes[7] * self.linear_scale
        
        # Axe horizontal gauche pour vitesse angulaire
        self.cmd_msg.angular.z = joy_msg.axes[3] * self.angular_scale
        
        # Publier la commande
        self.cmd_pub.publish(self.cmd_msg)

if __name__ == '__main__':
    try:
        node = TeleopNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass