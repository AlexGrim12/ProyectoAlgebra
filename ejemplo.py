import numpy as np

# Crear una matriz para representar los datos de entrenamiento
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([7, 8, 9])

# Crear una matriz para almacenar los pesos
w = np.array([0, 0])

# Realizar una regresión lineal simple
for i in range(1000):
    y_pred = X.dot(w)
    error = y_pred - y
    grad = X.T.dot(error) / len(X)
    w -= 0.01 * grad

print(w)

