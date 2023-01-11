import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Cargar los datos de entrenamiento y de prueba
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar los datos de entrenamiento y de prueba
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Aplanar las imágenes de 28x28 a un vector de 784 elementos
X_train = X_train.reshape((X_train.shape[0], 784))
X_test = X_test.reshape((X_test.shape[0], 784))

# Convertir las etiquetas a categorías
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(784,)))
model.add(Dense(10, activation="softmax"))

# Compilar el modelo
model.compile(optimizer="adam", loss="categorical_cross", metrics=["accuracy"])

