# Manejo de Codigo Paquete kidnaped_robot
---
## Setup del Paquete
---
Como se mencionó en anteriormente esta carpeta de trabajo es un puede descargarse e instalarse en un Workspace de ROS de la distribución Noetic.
1. Por lo tanto para poder trabajar con los archivos de este paquete se necesita primero tener un distribución de ROS Noetic activa y con un workspace ya generado para instalar este paquete y realizar un `catkin_make` sobre el workspace para su correcta integración. 
2. Instalar paquetería de Jackal UGV - Ya que este paquete trabaja con un Jackal UGV de la empresa clearpath es necesario descargar los paquetes auxiliares para un correcto funcionamiento de este paquete. Puede seguir el tutorial en el siguiente link [Clearpath Jackal](http://www.clearpathrobotics.com/assets/guides/noetic/jackal/simulation.html) 


## Trabajar con los codigos
---
### Lanzar simulación Gazebo
Para lanzar la simulación en gazebo y poder mover el jackal de manera correcta se tienen que seguir los siguiente pasos:
1. Lanzar la simulación en el software de Gazebo con el comando `roslaunch kidnaped_robot gazebo_greenhouse.launch`
2. Lanzar cualquier nodo de navegación con el comando `roslaunch jackal_navigation odom.launch`
3. Lanzar nodo de visualización Rviz con el comando `roslaunch jackal_viz view_robot.launch config:=navigation`

### Lanzar simulación principal Gazebo
Para lanzar la simulación en gazebo y poder utilizar la implementación de switch entre sistemas de localización debe seguir lo siguientes pasos:
1. Lanzar la simulación en el software de Gazebo con el comando `roslaunch kidnaped_robot gazebo_greenhouse.launch`
2. Lanzar todos los nodos con el siguiente comando `roslaunch kidnaped_robot prueba_gazebo.launch`
3. Lanzar el nodo de información con el siguiente comando `rosrun kidnaped_robot infoNode.py`

### Notas
Algunos paths dentros los programas de infoNode.py y LeNet.py deben modificarse para un correcto funcionamiento de los mismo.
