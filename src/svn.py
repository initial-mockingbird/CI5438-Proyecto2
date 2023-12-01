
import numpy as np
import pandas as pd
from sklearn import svm
from src.iris import iris_binary, resultados


def cont(train,train_t,test,test_t,targets,learning_rate=None):
  
  clf = svm.SVC()
  clf.fit(train, np.ravel(train_t))
  predictions = clf.predict(test)
  delta = predictions - np.ravel(test_t)
  df = pd.DataFrame(delta, columns=targets)

  return resultados(df,targets,learning_rate,output="")


def cont_linear(train,train_t,test,test_t,targets,learning_rate=None):
  
  clf = svm.SVC(kernel="linear")
  clf.fit(train, np.ravel(train_t))
  predictions = clf.predict(test)
  delta = predictions - np.ravel(test_t)
  df = pd.DataFrame(delta, columns=targets)

  return resultados(df,targets,learning_rate,output="")



def svn(path):
  def max(xs):
    try:
      return list(map(np.max,xs))
    except:
      return xs
  def min(xs):
    try:
      return list(map(np.min,xs))
    except:
      return xs
  def avr(xs):
    try:
      return list(map(np.average,xs))
    except: 
      return xs
  results_single_setosa = iris_binary(path,["species:Iris-setosa"])(cont)
  results_single_versicolor = iris_binary(path,["species:Iris-versicolor"])(cont)
  results_single_virginica  = iris_binary(path,["species:Iris-virginica"])(cont)

  aux = [ (target,acc,p,r,falso_positivo,falso_negativo) for (target,acc,p,r,falso_positivo,falso_negativo) in results_single_setosa]
  targets    =  pd.DataFrame(aux,columns=["target","acc","p","r","falso_positivo","falso_negativo"])
  grouped = targets.groupby(by=["target"])
  grouped.agg([max,min,avr]).to_csv("./results/svn_setosa.csv")

  aux = [ (target,acc,p,r,falso_positivo,falso_negativo) for (target,acc,p,r,falso_positivo,falso_negativo) in results_single_versicolor]
  targets    =  pd.DataFrame(aux,columns=["target","acc","p","r","falso_positivo","falso_negativo"])
  grouped = targets.groupby(by=["target"])
  grouped.agg([max,min,avr]).to_csv("./results/svn_versicolor.csv")

  aux = [ (target,acc,p,r,falso_positivo,falso_negativo) for (target,acc,p,r,falso_positivo,falso_negativo) in results_single_virginica]
  targets    =  pd.DataFrame(aux,columns=["target","acc","p","r","falso_positivo","falso_negativo"])
  grouped = targets.groupby(by=["target"])
  grouped.agg([max,min,avr]).to_csv("./results/svn_virginica.csv")


  results_single_setosa_linear = iris_binary(path,["species:Iris-setosa"])(cont)
  results_single_versicolor_linear = iris_binary(path,["species:Iris-versicolor"])(cont)
  results_single_virginica_linear  = iris_binary(path,["species:Iris-virginica"])(cont)

  aux = [ (target,acc,p,r,falso_positivo,falso_negativo) for (target,acc,p,r,falso_positivo,falso_negativo) in results_single_setosa_linear]
  targets    =  pd.DataFrame(aux,columns=["target","acc","p","r","falso_positivo","falso_negativo"])
  grouped = targets.groupby(by=["target"])
  grouped.agg([max,min,avr]).to_csv("./results/svn_setosa_linear.csv")

  aux = [ (target,acc,p,r,falso_positivo,falso_negativo) for (target,acc,p,r,falso_positivo,falso_negativo) in results_single_versicolor_linear]
  targets    =  pd.DataFrame(aux,columns=["target","acc","p","r","falso_positivo","falso_negativo"])
  grouped = targets.groupby(by=["target"])
  grouped.agg([max,min,avr]).to_csv("./results/svn_versicolor_linear.csv")

  aux = [ (target,acc,p,r,falso_positivo,falso_negativo) for (target,acc,p,r,falso_positivo,falso_negativo) in results_single_virginica_linear]
  targets    =  pd.DataFrame(aux,columns=["target","acc","p","r","falso_positivo","falso_negativo"])
  grouped = targets.groupby(by=["target"])
  grouped.agg([max,min,avr]).to_csv("./results/svn_virginica_linear.csv")