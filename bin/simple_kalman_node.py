#!/usr/bin/env Python

import rospy

if __name__ == '__main__':
    rospy.loginfo("Initialising simple_kalman_node")
    rospy.init_node("simple_kalman_node")
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
