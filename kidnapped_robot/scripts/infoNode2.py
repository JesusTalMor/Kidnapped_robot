#!/usr/bin/env python3 
import os

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class InfoNode():
  ''' This class colects position data from both positioning methods'''
  def __init__(self): 
    # Callback fuction when the node is turned off
    rospy.on_shutdown(self.cleanup)  
    ###******* VARIABLES **********###
    rate = 1 # One image per second
    self.info_list = []
    self.SWITCH = False
    ###******* INIT PUBLISHERS *******### 
    # TODO Publicar pose o coordenadas en /final_pos
    ###******* INIT SUSCRIBERS *******### 
    rospy.Subscriber('/switch', Bool, self.NET_flag)
    #********** INIT NODE **********### 
    r = rospy.Rate(rate) #1Hz 
    rospy.loginfo("Node initialized " + str(rate)+ " hz")
    while not rospy.is_shutdown():
      # os.system('clear')
      self.info_list.append(self.SWITCH)
      r.sleep()  
  
  def NET_flag(self, data:Bool):
    # Sacar el booleano de la red
    self.SWITCH = data.data

  def cleanup(self):
    # Tomar la lista de las poses y guardarla en un Txt
    report_name = 'report_LeNet_2pasillos.txt'
    folder_path = f'/home/fritzfate/catkin_ws/src/kidnapped_robot/reports/{report_name}'
    report_txt_writer = open(folder_path, 'w', encoding="utf-8")
    report_txt_writer.write(f'Estado de Switch \n')
    for status in self.info_list:
      report_txt_writer.write(f'{status} \n')
    report_txt_writer.close()


############################### MAIN PROGRAM #################################### 
if __name__ == "__main__": 
    rospy.init_node("Information_Node", anonymous=True) 
    try:
        InfoNode()
    except rospy.ROSInterruptException:
        rospy.logwarn("EXECUTION COMPLETED SUCCESFULLY")
