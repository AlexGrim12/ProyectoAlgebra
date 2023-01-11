import numpy as np
#crear una matriz vacia:
A = np.array([])
#crear una matriz a partir de una lista:
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#Crear una matriz llena de ceros o unos:
C = np.zeros((3, 3))
D = np.ones((3, 3))
#Crear una matriz con un valor específico:
E = np.full((3, 3), 7)
#Crear una matriz identidad:
F = np.eye(3)
#Crear una matriz con valores aleatorios:
G = np.random.rand(3, 3)
H = np.random.randn(3, 3)
#Crear una matriz a partir de una secuencia:
I = np.arange(0, 9).reshape(3,3)
