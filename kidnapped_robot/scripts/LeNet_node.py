#!/usr/bin/env python3 
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch
import torch.nn as nn
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.relu = nn.ReLU()
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5,5), stride=(1,1), padding=(0,0))
    self.bn1 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0))
    self.bn2 = nn.BatchNorm2d(16)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=(1,1), padding=(0,0))
    self.bn3 = nn.BatchNorm2d(32)
    self.linear1 = nn.Linear(4608, 83)
    self.linear2 = nn.Linear(83,2)
  
  def forward(self,x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.pool(x)
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.pool(x)
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.relu(self.linear1(x))
    x = self.linear2(x)
    return x


class NeuralNode():
  ''' This class colects position data from both positioning methods'''
  def __init__(self): 
    
    rospy.on_shutdown(self.cleanup)  # Funcion para cuando el nodo termina
    
    ###******* LENETMODEL *********###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_path = 'LENET_300E_ROT.pth'
    # model_path = '/home/fritzfate/catkin_ws/src/kidnapped_robot/scripts/LENET_500E.pth'
    model_path = '/home/fritzfate/catkin_ws/src/kidnapped_robot/scripts/LENET_300E_ROT.pth'
    LENET_model = LeNet().to(device=device)
    LENET_model.load_state_dict(torch.load(model_path))
    LENET_model.eval()
    
    ###******* VARIABLES **********###
    rate = 1 # One image per second
    self.state = None
    modify_flag = True

    self.image_received = 0
    self.pose_received = 0
    
    ###******* INIT PUBLISHERS *******### 
    self.pub_state = rospy.Publisher('/switch', Bool, queue_size=1)
    self.pub_pose = rospy.Publisher('/set_pose', PoseWithCovarianceStamped, queue_size=1)
    
    ###******* INIT SUSCRIBERS *******### 
    self.bridge = CvBridge()
    rospy.Subscriber('laser_image', Image, self.image_cb)
    # rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.MCL_cb)
    
    #********** INIT NODE **********### 
    r = rospy.Rate(rate) #1Hz 
    rospy.loginfo("Node initialized " + str(rate)+ " hz")
    while not rospy.is_shutdown():
      os.system('clear')
      if not self.image_received: continue # Checar si recibimos imagen
      
      main_image = self.cv_image # Pasamos la imagen forma 128 x 128 x 1
      # * Prediccion del Estado
      image_net = torch.tensor(main_image).to(device=device, dtype=torch.float).view(1, 1, 128, 128)
      _, pred = LENET_model(image_net).max(1)
      
      robot_status = Bool() # Status del Robot (Dentro o Fuera del Pasillo) : Default False
      robot_status.data = False
      # * Dentro del Pasillo
      if pred[0] == 1:   
        rospy.logwarn('Pasillo Detectado')
        robot_status.data = 1
      
      # * Fuera del Pasillo
      else: 
        rospy.loginfo('Caso Normal')
      
      # * Publicar Estatus del Robot
      self.pub_state.publish(robot_status)
      self.image_received = 0
      self.pose_received = 0
      r.sleep()


  def image_cb(self, data:Image):
    # Tomar la image y transformarla para la Red
    if not self.image_received:   
      try: 
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        self.cv_image = cv2.resize(self.cv_image, (128,128))  
        self.image_received = 1 #Turn the flag on 
      except CvBridgeError as e: 
        print(e) 


  def MCL_cb(self, data:PoseWithCovarianceStamped):
    # Guardar la pose del robot usando MCL
    if not self.pose_received:
      self.MCL_pose = data
      self.MCL_pose.header.frame_id = 'odom'
      self.pose_received = 1


  def cleanup(self):
    pass

############################### MAIN PROGRAM #################################### 
if __name__ == "__main__": 
    rospy.init_node("Neural_Node", anonymous=True) 
    try:
        NeuralNode()
    except rospy.ROSInterruptException:
        rospy.logwarn("EXECUTION COMPLETED SUCCESFULLY")
