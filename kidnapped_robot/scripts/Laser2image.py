#!/usr/bin/env python3 
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan


class LaserScanToImage():
  ''' This class converts LaserScan reads to PNG Images'''
  def __init__(self): 
    # Callback fuction when the node is turned off
    rospy.on_shutdown(self.cleanup)  
    ###******* CONSTANTS **********###
    rate = 60 # One image per second
    self.lidar_data = 0
    self.img_count = 0
    self.scale = 0.04 # Escala es 1 pixel equivale a 0.5 metros
    len_x = 200
    len_y = 200
    self.desfase = 100 * 0.04
    lidar_values = 0
    # robot_pos = (100,200)
    ###******* INIT PUBLISHERS *******### 
    self.pub_image = rospy.Publisher('laser_image', Image, queue_size=1)
    ###******* INIT SUSCRIBERS *******### 
    self.bridge = CvBridge()
    rospy.Subscriber('/front/scan',LaserScan,self.scan_cb)
    # rospy.Subscriber('/scan',LaserScan,self.scan_cb)
    #********** INIT NODE **********### 
    r = rospy.Rate(rate) #1Hz 
    rospy.loginfo("Node initialized " + str(rate)+ " hz")
    while not rospy.is_shutdown():
      # Dentro de esta zona visualizamos la imagen
      # print(self.lidar_data)
      # Generamos una imagen de 200 pixeles por 200 pixeles
      
      image = np.full((len_x,len_y), 255)
      lidar_values = self.lidar_data
      if lidar_values != 0:
        image = self.create_image(lidar_values, image)
        image = image.astype(np.uint8)
        # publicar la imagen
        rospy.loginfo('Public Image')
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(image, "mono8"))
      r.sleep()
  
  def scan_cb(self, data:LaserScan):
    # Tenemos que pasar los datos del LaserScan a una variable para que no se mueva  
    temp_lidar_data = data.ranges
    angle_increm = data.angle_increment
    min_angle = data.angle_min 
    # print(min_angle)
    # print(data.angle_max)
    self.lidar_data = []
    # Vamos a tomar ahora mismo solo los datos del Lidar de 0 hasta 180
    for idx, value in enumerate(temp_lidar_data):
      main_angle = min_angle + (angle_increm * idx)
      # if value <= 8 and not np.isinf(value) and abs(main_angle) > (abs(min_angle)/2) :
      if value <= 8 and not np.isinf(value) and abs(main_angle) < (abs(min_angle)/2) :
        out = (main_angle, value)
        self.lidar_data.append(out)
  
  
  def create_image(self, lidar_values, image):

      for values in lidar_values:
        angle, dist = values
        x_coord = (np.sin(angle) * dist) + self.desfase
        # x_coord = (np.sin(angle) * -dist) + self.desfase
        # * Checar si se sale de los tamaños normales
        if x_coord < 0 or x_coord > 8: continue
        # y_coord = (np.cos(angle) * -dist)
        y_coord = (np.cos(angle) * dist)
        # * Checar si se sale de los tamaños normales
        if y_coord < 0 or y_coord > 8: continue
        
        x_coord_ind = int(x_coord/self.scale)
        y_coord_ind = int(y_coord/self.scale)

        # * Actualizar los datos
        image[y_coord_ind][x_coord_ind] = 0
      
      return image

  def cleanup(self):
    rospy.logwarn("Finalizando Nodo CORRECTAMENTE")

############################### MAIN PROGRAM #################################### 
if __name__ == "__main__": 
    rospy.init_node("LaserToImage", anonymous=True) 
    try:
        LaserScanToImage()
    except rospy.ROSInterruptException:
        rospy.logwarn("EXECUTION COMPLETED SUCCESFULLY")