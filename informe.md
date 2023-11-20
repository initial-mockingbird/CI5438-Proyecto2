# Daniel Pinto 15-11139, Pedro Rodriguez 15-11264

## Detalles de la implementacion.

La implementacion es _casi_ una copia 1 a 1 de lo visto en clase en forma matricial:

$$
\begin{align*}
\vec w_{i+1} &= \vec w_i + a \cdot \dfrac{d\ MSE}{d\ \vec w_i} \\
&= \vec w_i - \dfrac{\alpha}{2m}\ \vec x^{\ T} \cdot (\vec x \cdot \vec w_i - \vec y)  
\end{align*}
$$

Adicionalmente, necesitamos una medida que determine el accuracy de nuestro modelo, para esto utilizamos el coeficiente de determinacion $R^2$ definido como:

$$
\begin{align*}
\bar y &= \dfrac{1}{n} \sum_{i=1}^n y_i\\
SS_{res} &= \sum_i (y_i - f_i)^2\\
SS_{tot} &= \sum_i (y_i - \bar y)^2\\
R^2 &= 1 - \dfrac{SS_{res}}{SS_{tot}}
\end{align*}
$$

Aqui, entre mas cercano a $1$ sea el $R^2$ mas preciso es el modelo, y entre mas cercano a 0, mas predice la media.

Como hemos dicho, el corazon de la implementacion es casi una copia 1 a 1 de esto, modulo inicializacion de valores, renombramiento de los dataframe y recoleccion de errores:


```python
import pandas as pd
import numpy as np

def linear_regression(data,objective,a=1e-3,max_iter=3e4, epsilon=1e-3):

  # remapeamos los nombres para poder incluir "w0" sin clashes
  column_mapping = dict([ (name,f"w{i+1}") if name != objective else (objective,objective)  for (i,name) in enumerate(data.columns)])
  
  # construimos: X,Y,W_0
  X_df = data.copy().rename(column_mapping).drop(objective,axis=1)
  X_df.insert(0,"w0",1)
  Y = data[objective].to_numpy()
  X = X_df.to_numpy()
  W = X_df.copy().head(1).to_numpy()[0]

  # seteando valores iniciales del loop y constantes
  current_iteration = 0
  error = epsilon + 1
  m = len(Y)

  # Para plotear los errores
  acc_errors = []

  # metodo iterativo
  while(current_iteration < max_iter and error > epsilon ):
    P   = X @ W
    delta = P - Y
    W = W - a/m * (X.transpose() @ delta)
    error = abs(1/(2*m) * np.transpose(delta) @ delta) 
    acc_errors.append(error)
    current_iteration += 1
  
  x = np.array([i for (i,_) in enumerate(acc_errors)])
  acc_errors = np.array(acc_errors)
  
  return (x,acc_errors,W)
```

Notemos adicionalmente que $w_0$ lo definimos como la primera fila del dataset. Esto no ocasiona problemas ya que tenemos un dataset de mas de 1000 observaciones.

## Preprocesamiento de datos

Seguimos al pie de la letra la mayoria de las recomendaciones dadas por el profesor para realizar el preprocesamiento:

- Los tipos de datos fueron leidos acorde a los archivos de configuracion situados en `./config`, usando `string, unsigned int, unsigned big int, unsigned small int, numeric` como tipos de datos (todos nullables, con **UNICO** valor nulo el string vacio: `""`). El mapeo de estos datos a `dtypes` se puede ver en `src/config/preprocessing.py`:

```python
DATA_MAPPER =\
  { Data_Types['unsigned int']       : pd.UInt32Dtype()
  , Data_Types['unsigned big int']   : pd.UInt64Dtype()
  , Data_Types['unsigned small int'] : pd.UInt8Dtype()
  , Data_Types['string']             : pd.StringDtype()
  , Data_Types['numeric']            : pd.Float32Dtype()
  }
```

- La estrategia para menejo de `null` fue la de `remove`, esto porque son pocas las observaciones que poseen este tipo de datos. Sin embargo, esto se puede configurar por
columna en los archivos de configuracion, asignandole a cada `fill_strategy` una de: `'REMOVE'|'AVERAGE'|'COMMON'`. La implementacion de esto esta en `src/config/preprocessing.py`:

```python
def remove_strategy(df, column):
  df = df[df[column].notna()]
  return df

def common_strategy(df, column):
  df[column].fillna(df[column].mode()[0],inplace=True)
  return df

def average_strategy(df, column):
  df[column].fillna(df[column].mean(),inplace=True)
  return df
```

- La estrategia de categorizacion es la dummy, codificando cada valor del dominio como una variable binaria. Esto no es configurable.
- Se decide normalizar todas las variables (despues de la categorizacion). Esto es configurable a nivel global, no por columna. 

- Se realiza un split 80/20 (configurable a nivel global) aleatorio.
- Se entrena con $\alpha = 10^{-3}$ y con toleracia $\epsilon = 2\cdot 10^{-3}$
## Verificacion del modelo

Para verificar que nuestros algoritmos se esten comportando de manera correcta, se decidio utilizar el polinomio: $f(w_1,w_2) = 1 + 20w_1 + 50w_2$ y generar $1e4$ observaciones aleatorias para entrenar a nuestro modelo (la generacion de estos datos se puede encontrar en `src/random_sample.py`, y el entranamiento en `src/run_all.py` en la funcion `run_all_test`).

<p align="center">
  <img src="./imgs/testing.PNG" alt="Sublime's custom image"/>
</p>

Al plotear los errores por iteracion, obtenemos una curva bastante estandar para las regresiones lineales.

Luego, utilizaremos 2 metricas para verificar la precision de nuestro modelo con este ejemplo. La primera siendo el coeficiente de determinacion $R^2=0.98644$ el cual es bastante cercano a 1. Indicando que el modelo se ajusta bien.

Al trabajar sobre polinomios de grado 1, en donde conocemos los coeficientes de cada uno. Podemos utilizar la definicion de norma $l2$ para el espacio de funciones para definir el error relativo:


$$
\begin{align*}
|f(x,y)| &= \sqrt{\int_{R^2} f(x,y)^2 dxdy } \\
\langle f,g \rangle &= \dfrac{|f-g|}{|g|}
\end{align*}
$$

Sin perdida de generalidad, dejemos que $R^2=\{(x,y)\ |\ 0\leq x,y \leq 1 \}$, ya que nuestros inputs estan normalziados. Y dado que:

$$
\begin{align*}
  f_{pred} &= 5.45 + 17.05w_1 + 44.54w_2\\
  g_{ref}  &= 1 + 20 w_1 + 50 w_2
\end{align*}
$$

Obtenemos un error relativo de:

$$
\begin{align*}
\langle f_{pred},g_{ref} \rangle &= \dfrac{\sqrt{\int_0^1 \int_0^1(-4.454 + 2.94 \cdot x + 5.455 \cdot y)^2dxdy}}{\sqrt{\int_0^1 \int_0^1 (1 + 20 x + 50 y)^2 dxdy}} \\
&= \dfrac{1.8075}{39.2135} \\
&= 0.046094389277848494
\end{align*}
$$

Lo cual es un error bastante bajo, el cual se podria pensar como un "4\%" entre ambas funciones. Esto finalmente se puede corroborar graficamente, al ver el plano que forman ambas funciones:

<p align="center">
  <img src="./imgs/testing_3d.PNG" alt="Sublime's custom image" width="150%" height="150%"/>
</p>

Todo esto nos lleva a concluir no solo que nuestra implementacion es la correcta, sino que el coeficiente de determinacion $R^2$ es una buena metrica para medir precision.

## Entrenamiento de la data


Como no se implemento un mecanismo de regularizacion, se decidio correr regresion sobre 3 conjunto de columnas distintos:

- `config.yaml`: El cual contiene: `Make, Year, Kilometer, Fuel Type, Transmission, Owner, Seating Capacity, Fuel Tank Capacity, Price`
- `make_year_kilo_seating_fuel_location.yaml`: El cual contiene: `Make, Year, Location, Kilometer, Seating Capacity, Fuel Tank Capacity, Price`
- `no_categorics.yaml`: El cual contiene: `Seating Capacity, Fuel Tank Capacity, Length, Width, Height, Price`

<p align="center">
  <img src="./imgs/iteration_mse_4.PNG" alt="Sublime's custom image"/>
</p>

Al ver el grafico de iteracion vs error, podemos observar que tenemos asintotas similares para los 3 conjuntos de columnas, difiriendo significativamente solo en el punto de inflexion. Lo interesante viene cuando analizamos los coeficientes de determinacion:

<p align="center">
  <img src="./imgs/r2_score.PNG" alt="Sublime's custom image"/>
   <figcaption>Coeficiente de determinacion para 1000 iteraciones</figcaption>
</p>

<p align="center">
  <img src="./imgs/r2_score_2.PNG" alt="Sublime's custom image"/>
   <figcaption>Coeficiente de determinacion para 10000 iteraciones</figcaption>
</p>

<p align="center">
  <img src="./imgs/r2_score_3.PNG" alt="Sublime's custom image"/>
   <figcaption>Coeficiente de determinacion para 50000 iteraciones</figcaption>
</p>

<p align="center">
  <img src="./imgs/r2_score_4.PNG" alt="Sublime's custom image"/>
   <figcaption>Coeficiente de determinacion para 100000 iteraciones</figcaption>
</p>

En el cual vemos un incremento significativo en los 3 conjuntos de columnas

Y aunque el modelo aun con 10000 iteraciones no brinda una buena precision (tan solo de 0.5, llegando a ser solo un buen predictor para la media), vemos que a medida que las iteraciones crecen, aun hay cambios, llegando a obtener 0.65 de accuracy, estancandose en este valor. Lo cual sugiere que no vale la pena iterar sobre 50000 iteraciones.

Otra conclusion que pudieramos derivar de este modelo, es que debido a que tanto `config.yaml` como `make_year_kilo_seating_fuel_location.yaml` poseen precision similar bajo iteraciones altas, la interseccion de sus columnas nos dara un modelo mas refinado con resultados similares. Mejorando los tiempos de corrida.