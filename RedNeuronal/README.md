# Guia para Utilizar RedNeuronal
---
## 1. Pasos previos
---
Para poder trabajar con esta carpeta son necesarios 2 aspectos importantes. No import si estas en Windows o Linux.
### Instalación de Pytorch.

Ya que toda la red fue montada en la plataforma de pytorch, es necesario utilizar este paquete para poder trabajar de la misma manera.
Se puede seguir el siguiente tutorial para instalar pytorch de manera sencilla. [Instalar Pytorch](https://pytorch.org/get-started/locally/)

### Descargar el set de datos para entrenamiento.

En caso de que quieras entrenar una nueva red neuronal es necesario descargar la siguiente carpeta [Imagenes](https://drive.google.com/file/d/1l0Hwp4iuC6g6pmr-L_hbt6QA2glrgVx-/view?usp=share_link)
Una vez descargado el set de datos debes mover esta carpeta dentro de la carpeta RedNeuronal.


## 2. Trabajar con los códigos
---

Una vez que realices los pasos previos ya puedes empezar a trabajar con los archivos de la red neuronal algunos ejemplos que puedes realizar se ven acontinuación

### Checar LeNet precargadas
Usando el archivo check_LENET.py podemos utilizar cualquiera de las redes ya dentro del programa como (LENET_300_ROT.pth) o (LENET_500E.pth) corriendo dicho programa enseñara algunas imagenes del dataset para evaluar el modelo

### Entrenar un modelo
Usando el archivo LeNet.py podemos generar y entrenar nuevos modelos LeNeT simplemente podemos seguir los comentarios dentro del programa y realizar y entrenar nuevos modelos
