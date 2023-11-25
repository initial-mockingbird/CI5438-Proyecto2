# Daniel Pinto 15-11139, Pedro Rodriguez 15-11264

## Detalles de la implementacion.

Nuestra implementacion hace distincion entre 3 tipos de capas:

- Capas de activacion: Estas son capas ocultas que unicamente aplican una funcion de activacion al input que se les pase. No poseen estado (peso o bias)
- Capas lineales: Estas capas son capas ocultas que guardan como estado una matriz de pesos y de bias para ajustarse, y su output no aplica funciones de activacion.
- Capa de salida de activacion: La capa de salida de activacion es una capa de activacion, pero con la salvedad de que tiene una funcion de costos, y adicionalemente el gradiente de bajada ya en terminos de la funcion de error + la funcion de activacion. Esto nos salva una regla de la cadena.

La razon por la cual hacemos estas 3 distinciones, es para tener un backward propagation mas organico: la forma que se calcula el gradiente de bajada en cada iteracion, es simplemente llamando a un metodo polimorfo. Cosa que no pasaria si juntaramos capas de activacion con capas lineales. Puesto que tendriamos que o encontrar una expresion cerrada para esa combinacion, o tendriamos que hacer el paso de activacion explicito:

```python
# configuracion actual: 3 tipos de capas
for layer in reversed(hidden_layers):
  Y = activations.pop()
  downstream_grad = layer.X_grad(Y=Y,upstream_grad=upstream_grad)
  X = activations[-1]
  W_grads.append(layer.W_grad(X=X,upstream_grad=upstream_grad))
  upstream_grad = downstream_grad

# configuracion no tomada: 2 tipos de capas
for layer in reversed(hidden_layers):
  Y = activations.pop()
  # activacion explicita, activacion fija (se puede evitar anadiendo un metodo)
  error_gradient = logistic_derivative(Y) * upstream_grad
  Y = activations.pop() # sigue siendo correcto? Es mas dificil de razonar
  downstream_grad = layer.X_grad(Y=Y,upstream_grad=upstream_grad)
  X = activations[-1] # sigue siendo correcto? Es mas dificil de razonar
  W_grads.append(layer.W_grad(X=X,upstream_grad=error_gradient))
  upstream_grad = downstream_grad
```

Adicionalmente, proveemos dos clases de activacion para la capa de salida:

- Logistica con log-loss para clasificadores binarios:

```python
class LogisticOutput(OutputLayer):
  def get_output(self,X):
    return 1 / (1 + np.exp(-X))

  def X_grad(self, Y,T, upstream_grad = None):
    return (Y - T) / Y.shape[0]

  def get_cost(self,Y ,T):
    # 1e-7 para evitar log(0)
    return - (Y * np.log(T + 1e-7) + (1-Y)*np.log(1-T + 1e-7)).sum() / Y.shape[0]
```

- Softmax con cross entropy para clasificadores multiclase:

```python
def softmax(X):
  # Softmax es prone a under/overflow
  # pero tiene la propiedad de que:
  # softmax(x) = softmax(x-C) para una C constante
  # entonces podemos normalizar el rango para evitar estos problemas.
  z = X - np.max(X,axis=-1,keepdims=True)
  return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


class SoftMaxOutput(OutputLayer):
  def get_output(self,X):
    return softmax(X)

  def X_grad(self, Y,T, upstream_grad = None):
    return (Y - T) / Y.shape[0]

  def get_cost(self,Y ,T):
    return - (T * np.log(Y)).sum() / Y.shape[0]
```

El profesor atento notara que ninguna de estas dos capas de salida usan perdida cuadratica. Esto es debido a que el MSE es una funcion no convexa en el caso de componerse con la funcion logistica, lo cual hace que la funcion se pueda quedar atascada en algun minimo local muy por encima del minimo global. Lo cual se pudiera mitigar entrenando la red varias veces con distintos pesos iniciales (haciendo mas larga y tardia la implementacion). Pero adicionalmente, requiere mas iteraciones para encontrar convergencia, puesto que la penalidad de MSE para missclassifications, es menor a la de log-loss. Veamos un ejemplo en donde el modelo predice un output: $1$, cuando el verdadero valor es $0.1$

$$
\begin{align*}
  (Predicted - Target)^2 &=  (1-0.1)^2 = 0.81 \\
  -(Predicted \cdot log(Target)) &= -(1 \cdot log(0.1)) = 2.3
\end{align*}
$$

Teniendo en cuenta que no tomamos la funcion de error, sino el gradiente, y lo utilizamos para multiplicar por regla de la cadena. Esta diferencia de valores se hace mucho mas significativa.

Como ultimo detalle de implementacion, utilizamos minibatches de tamano 25 para la fase de entrenamiento:

```python
def _minibatch(self):
  nb_of_batches = self.X_train.shape[0] // self.batch_size
  return list(zip(
    np.array_split(self.X_train, nb_of_batches, axis=0),
    np.array_split(self.T_train, nb_of_batches, axis=0)))
```

## Clasificador binario: IRIS


Para los clasificadores binarios se siguio la regla de 80/20, sin prestar mucha atencion a otros detalles como lo son el balance de clases al hacer la separacion. Esto porque el split se hace de manera aleatoria para evitar cualquier sezgo que se pueda introducir al particionarlos a mano.


El threshold utilizado fue de $80%$, es decir que cualquier cualquier respuesta mayor a $0.8$ se considera que pertenece a la categoria.

## Clasificador binario: Sin capas ocultas

<p align="center">
  <img src="./imgs/sc_setosa.PNG.PNG" alt="Setosa error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario: SETOSA</figcaption>
</p>

<p align="center">
  <img src="./imgs/sc_versicolor.PNG.PNG" alt="Versicolor error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario: Versicolor</figcaption>
</p>

<p align="center">
  <img src="./imgs/sc_virginica.PNG.PNG" alt="Virginica error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario: Virginica</figcaption>
</p>

Al plotear las curvas de aprendizaje, vemos un comportamiento asintoticamente similar para cada clasificador. Los datos de estos resultados se pueden encontrar en los archivos `./results/sc_[setosa|versicolor|virginica].csv` en donde se muestran los resultados agrupados por learning rate. 

En particular, observamos que el clasificador para setosa posee un accuracy (definido como $\frac{acertados}{totales}$) perfecto con $100\%$ para cualquier learning rate superior a `1e-3`, el de versicolor logra solo un accuracy de $33.33\%$ para cualquier learning rate superior a `1e-3`, y para virginica un accuracy de $76.66\%$ para cualquier learning rate superior a `2e-1`.

Son estos resultados buenos? Para este problema en particular diriamos que si. Ya que el clasificador mas debil lo podemos descartar y trabajar con los dos clasificadores binarios que poseen un accuracy decente. 

Que el conjunto sea linealmente separable tiene implicaciones en esta topologia? Realmente no para esta topologia. Que el conjunto sea linealmente separable solo significa que no hace falta añadir capas ocultas, lo que hace el proceso de entrenamiento potencialmente menos costoso. Es decir, nos quita una dimension a la hora de resolver el problema. Ahora solo necesitamos variar el learning rate junto al numero de iteraciones.

## Clasificador binario: Una capa oculta


Para este clasificador se utilizo una capa oculta de 5 neuronas, esto debido a que los datos son linealmente separables, y se hipotiza que al proyectarlas en un espacio con mayores dimensiones, esta caracteristica se pronuncie mas.

<p align="center">
  <img src="./imgs/sc_l1_setosa.PNG.PNG" alt="Setosa error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con una capa oculta: setosa</figcaption>
</p>

<p align="center">
  <img src="./imgs/sc_l1_versicolor.PNG.PNG" alt="Versicolor error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con una capa oculta: Versicolor</figcaption>
</p>

<p align="center">
  <img src="./imgs/sc_l1_virginica.PNG.PNG" alt="Virginica error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con una capa oculta: Virginica</figcaption>
</p>

Los resultados de estas topologias, se encuentran en `results/sc[setosa|versicolor|virginica]_l1.csv`. Igualmente, agrupados por learning rate.

En el csv, encontramos accuracy similares: $100\%$ en setosa para learning rates mayores a $0.5$ y $80\%$ en Versicolor,Virginica, para learning rates mayores a $0.1$. Lo cual es una mejora estricta sobre nuestro anterior clasificador. Pero es una mejora real? 

Un $7\%$ de mejora es algo significativo, sin embargo, hay que notar que aunque el clasificador de Versicolor mejoro su accuracy en $40\%$, hubieramos tenido el mismo resultado si simplemente hubieramos corrido los otros dos clasificadores y luego haciendo descarte.

Es esta topologia suficientemente buena? Si! $80\%$ de precision es un buen resultado. 

## Clasificador binario: Dos capas ocultas

Finalmente para el clasificador binario de dos capas, utilizamos 5 neuronas en cada una de las capas. Tratando de mantener la dimensionalidad mayor a la del problema original, sin necesidad de irnos a espacios mas grandes.

<p align="center">
  <img src="./imgs/sc_l2_setosa.PNG.PNG" alt="Setosa error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con dos capas ocultas: Setosa</figcaption>
</p>

<p align="center">
  <img src="./imgs/sc_l2_versicolor.PNG.PNG" alt="Versicolor error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con dos capas ocultas: Versicolor</figcaption>
</p>

<p align="center">
  <img src="./imgs/sc_l2_virginica.PNG.PNG" alt="Virginica error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con dos capas ocultas: Virginica</figcaption>
</p>


Los resultados de estas topologias, se encuentran en `results/sc[setosa|versicolor|virginica]_l2.csv`. Igualmente, agrupados por learning rate.

Y para sorpresa de absolutamente nadie. Esta red converge con los mismos resultados que la de una capa. Esto es debido a que el conjunto de funciones que puede aproximar una red de dos capas, contiene al conjunto de funciones que puede aproximar una red de una sola. Y dado que los datos son linealmente separables, y por ende, pertenecen al conjunto de funciones que puede aproximar una red de una capa. Deberiamos tener resultados similares.

## Finalmente... Sobre clasificadores binarios

Argumentamos que como el conjunto es linealmente separable, entonces podemos realizar descarte. Sin embargo la verdadera pregunta era si nuestros resultados corroboran esto. Y la triste verdad es que... Si! Que la red de 0 capas ocultas tenga un accuracy bajo solo para la ultima clase es indiferente, puesto que de igual forma trabajamos sobre un universo cerrado!

## Clasificador multiclase

El clasificador multiclase, a diferencia del clasificador binario, hace uso de la funcion softmax en la capa de salida, usando cross entropy como funcion de error. En teoria, tenemos un par de condiciones extremadamente deseables para poder utilizar softmax: cada punto solo puede tener una unica clasificacion y un  conjunto linealmente separable. Sin embargo, como veremos a continuacion,  los resultados obtenidos terminan siendo subpar.

Los archivos para ambos clasificadores se encuentran en `./results/mc_1c.csv` y en  `./results/mc_2c.csv`

## Clasificador multiclase: Con una capa oculta

El clasificador con una sola capa oculta, de 4 neuronas, solo logra un accuracy de $35\%$ en el mejor de las clases,  fracasando en todas las predicciones del conjunto de setosa.

<p align="center">
  <img src="./imgs/mc_1c_acc.PNG.PNG" alt="Clasificador multiclase de 1 capa: error por iteracion"/>
   <figcaption>Accuracy para el clasificador multiclase con una capa oculta</figcaption>
</p>

<p align="center">
  <img src="./imgs/mc_1c.PNG.PNG" alt="Clasificador multiclase de 1 capa: error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador multiclase con una capa oculta</figcaption>
</p>

## Clasificador multiclase: Con dos capas ocultas

Aunque este clasificador de 4,5 neuronas, plantea una mejora sobre el anterior, aun los resultados poseen un accuracy bajo, .

<p align="center">
  <img src="./imgs/mc_2c_acc.PNG.PNG" alt="Clasificador multiclase de 1 capa: error por iteracion"/>
   <figcaption>Accuracy para el clasificador multiclase con una capa oculta</figcaption>
</p>

<p align="center">
  <img src="./imgs/mc_2c.PNG.PNG" alt="Clasificador multiclase de 1 capa: error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador multiclase con dos capas ocultas</figcaption>
</p>


## Clasificador multiclase: Con una capa oculta: Logistic

Y aunque cambiemos nuestro clasificador de una capa con 5 neuronas, a que use como funcion de salida, la funcion logistica. Obtenemos resultados aun mas bajos que los usados con la funcion softmax.

<p align="center">
  <img src="./imgs/mc_1c_log_acc.PNG.PNG" alt="Clasificador multiclase de 1 capa: error por iteracion"/>
   <figcaption>Accuracy para el clasificador multiclase con una capa oculta</figcaption>
</p>

<p align="center">
  <img src="./imgs/mc_1c_log.PNG.PNG" alt="Clasificador multiclase de 1 capa: error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador multiclase con dos capas ocultas</figcaption>
</p>

## Por que tan salado?

Realmente, no es culpa de la funcion de activacion usada en la capa de salida, puesto que impuso una mejora sobre la funcion logistica. Tampoco es problema de la topologia ni de los hiperparametros puesto que son los mismos o similares a la de los clasificadores binarios. Entonces, donde esta el problema?

Lo ultimo que queda por inspeccionar es overfitting, o falta de datos. Nosotros nos inclinamos por lo 2do. Los clasificadores multiclase necesitan alimentarse de mas informacion, puesto  que este trata de comparar caracteristicas mas complejas (interacciones entre las clases?) que los clasificadores binarios. Y el cuerpo con el que trabajamos, es bastante pequeno, tan solo 150 muestras.


## Clasificador de spam.

## Preprocesamiento de los datos.

Únicamente se normalizaron los datos. Inicialmente se utilizó normalización por varianza y media, sin embargo esto originó overflows constantemente en la función de activación logística, por lo que se utilizó normalización por mínimo y máximo.

## Experimentos realizados.

Para todos los experimentos realizados, se siguió la regla 70/30 y el threshold utilizado fue de $50%$, es decir que cualquier cualquier respuesta mayor a $0.5$ se considera que el correo inspeccionado es spam.

En total se realizaron 18 experimentos, en donde se varió la topología de la red, el learning rate y la hipótesis utilizada, siendo entrenado cada modelo a lo largo de 5000 iteraciones.

### Topologías de la red.

Se utilizaron las siguientes topologías de red:

- Sin capas ocultas.
- Una sola capa oculta de 5 neuronas.
- Dos capas ocultas, la primera de 5 neuronas y la segunda de 2.

Siendo arbitraria la elección de neuronas por capa para estas últimas 2 topologías. Cada una de estas topologías tiene una neurona de salida con activación logística y capas ocultas utilizando esta misma función.

### Learning rate.

Se utilizaron los siguientes valores para el learning rate:

- $0.1$
- $0.01$
- $0.001$

### Hipótesis utilizada.

Se utilizaron dos hipótesis: La primera conteniendo todos los parámetros de entrada ofrecidos por el dataset recibido, y la segunda utilizando una cantidad selecta de estos. Esta hipótesis reducida, contiene los siguientes parámetros:

- word_freq_address
- word_freq_internet
- word_freq_order
- word_freq_mail
- word_freq_receive
- word_freq_free
- word_freq_business
- word_freq_email
- word_freq_credit
- word_freq_money
- word_freq_original
- word_freq_project
- char_freq_!
- char_freq_$
- capital_run_length_average
- capital_run_length_longest
- capital_run_length_total

Se redujo la cantidad de parámetros de entrada debido a que al examinar todos los disponibles, algunos aparentan no tener relación alguna con la clasificación de correos spam, i.e: _word_freq_george, word_freq_650, word_freq_hp, word_freq_1999, word_freq_pm._

## Resultados

Para cada experimento se almacenó el error en cada iteración, y adicionalmente se calcularon las métricas _Precision_ y _Recall_ para la evaluación del rendimiento de las redes, siendo estas dos definidas como:

$PRECISION = {True\_Positives \over {(True\_Positives + False\_Positives)}}$

$RECALL = {True\_Positives \over {(True\_Positives + False\_Negatives)}}$

Donde _Precision_ nos ayudará a medir que proporción de nuestras predicciónes positivas eran realmente positivas, mientras que el _Recall_ nos ayudará a medir que proporción de resultados verdaderamente positivos logra identificar.

En la siguientes secciones se presentan los gráficos y métricas resultantes en cada experimento para ambas hipótesis. Acotamos que las gráficas obtenidas para los experimentos con una topología de 2 capas no presentan una coloración correcta, sin embargo aclaramos que en ambas gráficas la curva con mayor error final representa un learning rate de $0.001$, la curva con el menor error un learning rate de $0.1$, y la restante $0.01$.

### Hipótesis completa

<p align="center">
  <img src="./imgs/spam_fig_ai_[].png" alt="Spam error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario sin capas ocultas y todos los datos de entrada</figcaption>
</p>

<p align="center">
  <img src="./imgs/spam_fig_ai_[4].png" alt="Spam error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con una capa oculta y todos los datos de entrada</figcaption>
</p>

<p align="center">
  <img src="./imgs/spam_fig_ai_[5 2].png" alt="Spam error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con dos capas ocultas y todos los datos de entrada</figcaption>
</p>

| Experimento                                                | Precision | Recall |
|------------------------------------------------------------|----------:|-------:|
| Hipótesis completa. Sin capas ocultas. Learning Rate 0.1   |     0.91  |   0.86 |
| Hipótesis completa. Sin capas ocultas. Learning Rate 0.01  |     0.90  |   0.83 |
| Hipótesis completa. Sin capas ocultas. Learning Rate 0.001 |     0.88  |   0.72 |
| Hipótesis completa. Una capa oculta. Learning Rate 0.1     |     0.92  |   0.91 |
| Hipótesis completa. Una capa oculta. Learning Rate 0.01    |     0.91  |   0.87 |
| Hipótesis completa. Una capa oculta. Learning Rate 0.001   |     0.88  |   0.77 |
| Hipótesis completa. Dos capas ocultas. Learning Rate 0.1   |     0.90  |   0.93 |
| Hipótesis completa. Dos capas ocultas. Learning Rate 0.01  |     0.91  |   0.91 |
| Hipótesis completa. Dos capas ocultas. Learning Rate 0.001 |     0.63  |  0.035 |

### Hipótesis reducida
<p align="center">
  <img src="./imgs/spam_fig_si_[].png" alt="Spam error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario sin capas ocultas y menos datos de reducida</figcaption>
</p>

<p align="center">
  <img src="./imgs/spam_fig_si_[4].png" alt="Spam error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con una capa oculta e hipótesis reducida</figcaption>
</p>


<p align="center">
  <img src="./imgs/spam_fig_si_[5 2].png" alt="Spam error por iteracion"/>
   <figcaption>Error por iteracion para el clasificador binario con dos capas ocultas e hipótesis reducida</figcaption>
</p>


| Experimento                                                | Precision | Recall |
|------------------------------------------------------------|----------:|-------:|
| Hipótesis reducida. Sin capas ocultas. Learning Rate 0.1   |     0.84  |   0.72 |
| Hipótesis reducida. Sin capas ocultas. Learning Rate 0.01  |     0.82  |   0.62 |
| Hipótesis reducida. Sin capas ocultas. Learning Rate 0.001 |     0.85  |   0.40 |
| Hipótesis reducida. Una capa oculta. Learning Rate 0.1     |     0.84  |   0.82 |
| Hipótesis reducida. Una capa oculta. Learning Rate 0.01    |     0.84  |   0.74 |
| Hipótesis reducida. Una capa oculta. Learning Rate 0.001   |     0.82  |   0.48 |
| Hipótesis reducida. Dos capas ocultas. Learning Rate 0.1   |     0.83  |   0.83 |
| Hipótesis reducida. Dos capas ocultas. Learning Rate 0.01  |     0.84  |   0.81 |
| Hipótesis reducida. Dos capas ocultas. Learning Rate 0.001 |      0.0  |    0.0 |

## Comparación de hipótesis
### Error por iteración.

Las gráficas para ambas hipótesis presentan curvas similares para topologías y learning rate equivalentes, donde aquellos experimento realizados con un mayor learning rate reducen con una mayor rapidez el error obtenido que sus contrapartes con un learning rate de menor valor, adicionalmente logrando reducir en mayor medida para las iteraciones finales. Algo a destacar es que para los experimentos realizados con la hipótesis completa, se obtuvo un menor error a comparación que el error obtenido al entrenar con la hipótesis reducida.

### Métricas.

En ambas hipótesis se puede observar que la métrica _Precision_ no varía significativamente en cada experimento, siendo la excepción a esto los experimentos realizados con 2 capas ocultas y un learning rate de $0.001$, los cuales presentan valores significativamente menores a sus contrapartes. Adicionalmente, también es notable como la métrica _Recall_ disminuye su valor junto al valor del learning rate.

Si comparamos ambas hipótesis y sus métricas, notamos que para todos los experimentos realizados con la hipótesis completa las métricas presentan un mayor valor que sus contrapartes realizadas con la hipótesis reducida.

### Qué hipótesis es mejor?

La idea de una hipótesis reducida surge tras examinar la descripción de cada columna de los datos de entrada, y llegar a la conclusión aparente de que no todos estos parámetros de entrada tienen sentido para el problema que queremos resolver, creando así la idea de que al eliminar posibles parámetros no relevantes para nuestro problema tendremos mejores resultados. 

Al comparar los resultados de los experimentos realizados entre ambas hipótesis, claramente la hipótesis completa presenta mejores resultados que su contraparte, lo cual parece tumbar la idea de que la hipótesis reducida es superior dado a la eliminación de parámetros no relevantes. Esto nos lleva a la conclusión de que la elección de una hipótesis es una tarea engañosa, dado a que podemos teorizar que una hipótesis nos va a dar mejores resultados debido a razones aparentemente lógicas, pero al momento de examinar el rendimiento de la red entrenada se demuestra lo contrario. Asi mismó, por más insignificante que pueda parecer un parámetro de entrada para resolver nuestro problema, vale la pena no descartarlo inmediatamente e intentar determinar mediante la experimentación su utilidad para la resolución de este.

## Comparación de cantidad de capas ocultas

Para ambas hipótesis se puede observar en las gráficas como el error de los experimentos con una mayor cantidad de capas ocultas, disminuye respecto a los experimentos análogos de la misma hipótesis pero con una cantidad menor de capas ocultas. La única excepción aparente a esta observación es el error obtenido en los experimentos con dos capas ocultas y un learning rate de $0.001$, posiblemente porque este valor no es lo suficientemente alto para obtener buenos resultados con una topología así.