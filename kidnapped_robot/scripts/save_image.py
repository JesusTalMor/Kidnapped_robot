#!/usr/bin/env python3 
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class ImageSaver():
  ''' This class converts LaserScan reads to PNG Images'''
  def __init__(self): 
    # Callback fuction when the node is turned off
    rospy.on_shutdown(self.cleanup)  
    ###******* CONSTANTS **********###
    rate = 1 # One image per second
    self.image_received = 0
    self.count = 0
    self.precount = 297 + 223
    ###******* INIT PUBLISHERS *******### 
    # ! No publisher
    ###******* INIT SUSCRIBERS *******### 
    self.bridge = CvBridge()
    rospy.Subscriber('laser_image',Image,self.image_cb)
    #********** INIT NODE **********### 
    r = rospy.Rate(rate) #1Hz 
    rospy.loginfo("Node initialized " + str(rate)+ " hz")
    while not rospy.is_shutdown():
      if self.image_received and self.count < 1000:
        main_image = self.cv_image
        rospy.loginfo('New Image')
        cv2.imwrite(f'case_{self.precount + self.count}.png', main_image)
        self.count += 1
        self.image_received = 0
      r.sleep()
  
  def image_cb(self, data:Image):
    ## This function receives a ROS image and transforms it into opencv format
    if not self.image_received:   
      try: 
        print("received ROS image, I will convert it to opencv") 
        # We select bgr8 because it is the OpenCV encoding by default 
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")  
        self.image_received = 1 #Turn the flag on 
      except CvBridgeError as e: 
        print(e) 
  

  def cleanup(self):
    rospy.logwarn("Finalizando Nodo CORRECTAMENTE")
    rospy.logwarn(f'Imagenes Generadas {self.count}')
############################### MAIN PROGRAM #################################### 
if __name__ == "__main__": 
    rospy.init_node("ImageSaver", anonymous=True) 
    try:
        ImageSaver()
    except rospy.ROSInterruptException:
        rospy.logwarn("EXECUTION COMPLETED SUCCESFULLY")
