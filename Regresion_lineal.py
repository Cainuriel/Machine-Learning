import numpy
import scipy
import matplotlib.pyplot as mpt 

from sklearn.datasets import load_boston

# cargamos el set de datos.
boston = load_boston()

#imprimimos la descripcion de los datos
#print(boston.DESCR)

# vamos a intentar predecir el valor medio de la vivienda con el dato del numero de habitaciones.

# cogemos todas las filas de la columna cinco. Son los inputs: 
# la media de habitaciones por barrio.
x = numpy.array(boston.data[:,5])
# los outputs son los valores medios de la vivienda. 
# que en este Dataset estan en el Target.
y = numpy.array(boston.target)
#creamos el eje de coordenadas con la libreria matplotlib
# el parametro alpha genera transparencia en la nube de datos
mpt.scatter(x,y,alpha = 0.3)
# se añade una columna de unos, tantos como datos hay, para el termino independiente:
# la T es para sacar la inversa de la matriz.
x = numpy.array([numpy.ones(506),x]).T
# a continuacion la formula que minimiza los cuadrados ordinarios y se marcara con una linea.
# ello se llama REGRESION LINEAL. aqui la hacemos matematicamente, pero las librerias ya la
#disponemos.
#la formula:
# BETA = (X^{T}X)^{-1}X^{T}Y
# numpty tiene las transpuestas con .T
# la arroba genera la multiplicacion matricial.
# para sacar la inversa de toda la ecuacion usamos linalg.inv()
beta = numpy.linalg.inv(x.T @ x) @ x.T @ y
# ahora vamos a dibujar la linea que nos muestre en la grafica
# los valores de progresion de los datos.
# queremos que la linea empiece en el valor 4 de x y acabe en el
# valor 9 de x. 
mpt.plot([4,9],[beta[0] + beta[1] * 4, beta[0] + beta[1] * 9], c = "red")
# finalmente se muestra con el metodo show
mpt.show()




