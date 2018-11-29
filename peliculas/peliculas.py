import pandas
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy 

regr = linear_model.LinearRegression()

datos = pandas.read_csv("movies2.csv")

dataframe = pandas.DataFrame(datos)
# vamos a comprobar si la valoracion del imdb corresponde a los likes en las peliculas.
x = dataframe['movie_facebook_likes']
y = dataframe['imdb_score']
# vamos a dar un formato de array a los datos.
arrayx = x[:,numpy.newaxis]
# lo comprobamos.
print(arrayx)
regr.fit(arrayx,y)
regr.coef_

m = regr.coef_[0]
b = regr.intercept_
y_predicion = m * arrayx + b # es el valor que se predice.
print("y = {0} * x + {1}".format(m,b))
print(regr.predict(arrayx)[0:5])

print("Valor de r^2: ",r2_score(y,y_predicion))
import matplotlib.pyplot as plt

plt.scatter(x,y,color = 'blue')
plt.plot(x,y_predicion, color = 'orange')
plt.title("Regresion Lineal.",fontsize = 16)
plt.xlabel("Likes de Facebook",fontsize = 13)
plt.ylabel("Valoracion IMDB",fontsize = 13)
plt.show()





