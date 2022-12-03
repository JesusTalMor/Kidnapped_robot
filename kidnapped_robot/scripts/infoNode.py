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
    self.list_info_method = []
    self.list_amcl = []
    self.list_odom = []
    self.SWITCH = None
    self.ODOM_pose = None
    self.AMCL_pose = None

    self.ODOM_flag = False
    self.AMCL_flag = False
    start_flag = True 

    ODOM_MEMO = (0,0)
    AMCL_MEMO = (0,0)

    ODOM_method_memo = (None,None)
    ###******* INIT PUBLISHERS *******### 
    # TODO Publicar pose o coordenadas en /final_pos
    ###******* INIT SUSCRIBERS *******### 
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.MCL_cb)
    rospy.Subscriber('/odometry/filtered', Odometry, self.ODOM_cb)
    rospy.Subscriber('/switch', Bool, self.NET_flag)
    #********** INIT NODE **********### 
    r = rospy.Rate(rate) #1Hz 
    rospy.loginfo("Node initialized " + str(rate)+ " hz")
    while not rospy.is_shutdown():
      os.system('clear')
      if self.ODOM_flag and ODOM_method_memo == (None,None):
        # Poner odometría incial
        ODOM_method_memo = self.ODOM_pose
      pose_x, pose_y = (None, None)
      if self.ODOM_flag: ODOM_ACTUAL = self.ODOM_pose
      else: continue
      if self.AMCL_pose: AMCL_ACTUAL = self.AMCL_pose
      else: continue
      
      if self.SWITCH:
        # * Caso Pasillo
        # rospy.loginfo(f'\nUSANDO ODOMETRIA')
        print('USANDO ODOMETRIA')
        if start_flag:
          # Guardamos la posicion de odometria y AMCL
            ODOM_MEMO = ODOM_ACTUAL
            AMCL_MEMO = AMCL_ACTUAL
            pose_x, pose_y = AMCL_MEMO
            start_flag = False
        else:
          # Actualizamos posicion con odometria previa
          DIFF_POS = self.tuple_operations(ODOM_ACTUAL, ODOM_MEMO, '-') # Diferencia con odometria de memoria
          pose_x, pose_y = self.tuple_operations(AMCL_MEMO, DIFF_POS, '+')
      else:
        # * Caso Fuera de Pasillo
        # rospy.loginfo(f'\nUSANDO AMCL')
        print('USANDO AMCL')
        pose_x, pose_y = AMCL_ACTUAL
        start_flag = True
      
      # * Publicar poses
      if pose_x != None:
        text = f'Coordenadas:[{pose_x:.2f},{pose_y:.2f}]'
        # rospy.loginfo(f'\n {text} \n')
        print(text)
        self.list_info_method.append((pose_x, pose_y))
        self.list_amcl.append(self.AMCL_pose)
        odom_to_list = self.tuple_operations(ODOM_ACTUAL, ODOM_method_memo, '-')
        self.list_odom.append(odom_to_list)

      self.ODOM_flag = False
      self.AMCL_flag = False
      r.sleep()  
  
  
  def NET_flag(self, data:Bool):
    # Sacar el booleano de la red
    self.SWITCH = data.data

  def ODOM_cb(self, data:Odometry):
    # Sacar posicion X y Y
    x_pos = data.pose.pose.position.x
    y_pos = data.pose.pose.position.y
    self.ODOM_pose = (x_pos, y_pos)
    self.ODOM_flag = True


  def MCL_cb(self, data:PoseWithCovarianceStamped):
    # Sacar posicion X y Y
    x_pos = data.pose.pose.position.x
    y_pos = data.pose.pose.position.y
    self.AMCL_pose = (x_pos, y_pos)
    self.AMCL_flag = True

  def tuple_operations(self, tuple1:tuple, tuple2:tuple, operand:str):
    # Descempacar los valores
    tuple1_x, tuple1_y = tuple1
    tuple2_x, tuple2_y = tuple2
    if operand == '+':
      res_x =  tuple1_x + tuple2_x
      res_y =  tuple1_y + tuple2_y
    elif operand == '-':
      res_x =  tuple1_x - tuple2_x
      res_y =  tuple1_y - tuple2_y
    return (res_x, res_y)

  def cleanup(self):
    # Tomar la lista de las poses y guardarla en un Txt
    report_name = 'report_info_route3_2.txt'
    folder_path = f'/home/fritzfate/catkin_ws/src/kidnapped_robot/reports/{report_name}'
    report_txt_writer = open(folder_path, 'w', encoding="utf-8")
    report_txt_writer.write(f'Methodo AMCL+ODOM \n')
    for (coord_x, coord_y) in self.list_info_method:
      report_txt_writer.write(f'{coord_x},{coord_y}\n')
    report_txt_writer.write(f'\n Methodo AMCL ONLY \n')
    for (coord_x, coord_y) in self.list_amcl:
      report_txt_writer.write(f'{coord_x},{coord_y}\n')
    report_txt_writer.write(f'\n Methodo ODOM ONLY \n')
    for (coord_x, coord_y) in self.list_odom:
      report_txt_writer.write(f'{coord_x},{coord_y}\n')
    report_txt_writer.close()


############################### MAIN PROGRAM #################################### 
if __name__ == "__main__": 
    rospy.init_node("Information_Node", anonymous=True) 
    try:
        InfoNode()
    except rospy.ROSInterruptException:
        rospy.logwarn("EXECUTION COMPLETED SUCCESFULLY")
